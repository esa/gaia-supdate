#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""
Module for solving systems of linear equations.

These methods are equivalent to the Java implementation in the Gaia AGIS processing software.

Note
----
credit authors that contributed to this code:
Sahlmann, Bombrun, Lindegren

"""
from collections import OrderedDict
import copy
from dataclasses import dataclass
import logging
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .metrics import SolutionStatistic, chi_squared


def decay_downweight(z: np.ndarray, scale_factor: float = 1.) -> np.ndarray:
    """Return weight.

    See gaia.cu3.agistools.algo.observations.DecayDownweight.downweight(double, double).

    Parameters
    ----------
    z : ndarray
        The input data array.
    scale_factor : float
        The scale factor

    Returns
    -------
    ndarray
        Array of decay downweights

    """
    a = 1.77373519609519
    b = 1.14161463726663
    x = np.abs(scale_factor * np.array(z))
    return np.piecewise(x,
                        [x <= 2,
                         (2 < x) & (x < 3),
                         x >= 3],
                        [1,
                         lambda x: 1 - a * (x - 2) ** 2 + b * (x - 2) ** 3,
                         lambda x: np.exp(-x / 3)])


def huber_downweight(z: np.ndarray, scale_factor: float = 1.) -> np.ndarray:
    """Return weight.

    See gaia.cu3.agistools.algo.observations.HuberDownweight

    Parameters
    ----------
    z : ndarray
        The input data array.
    scale_factor : float
        The scale factor

    Returns
    -------
    ndarray
        Array of Huber downweights

    """
    c = 2.0
    x = np.abs(scale_factor * np.array(z))
    return np.piecewise(x,
                        [x <= c,
                         x > c],
                        [1,
                         lambda x: c * (2 * x - c) / x ** 2])


# Parameters of the AGIS solver
# cf. gaia.cu3.agistools.algo.gis.source.SourceUpdateStatistics.Options
AGIS_SOLVER_OPTIONS = {}
AGIS_SOLVER_OPTIONS['downweighter'] = decay_downweight
AGIS_SOLVER_OPTIONS['downweightLimit'] = 0.2
AGIS_SOLVER_OPTIONS['tolEsvExcess'] = 1e-6
AGIS_SOLVER_OPTIONS['maxIterExcess'] = 10
AGIS_SOLVER_OPTIONS['dfWeight'] = 1.0
AGIS_SOLVER_OPTIONS['maxIterRobust'] = 30
AGIS_SOLVER_OPTIONS['tolEsvRobust'] = 1e-6


def estimate_excess_noise(residuals, sigmas, weights, nu):
    """Estimate the excess noise from given residuals.

    See Equation 64 of Lindegren at al. 2012, A&A, 538, A78.

    Parameters
    ----------
    residuals: narray
        Array of residuals.
    sigmas: narray
        Array of sigmas.
    weights: narray
        Array of weights.
    nu: int
        Number of degrees of freedom.

    Returns
    -------
    narray
        Array containing the excess noise estimation.

    """
    y = 0
    sy = np.sum(weights * residuals ** 2 / (sigmas ** 2))
    if sy < nu:
        return 0
    else:
        for i in range(0, 3):
            dsdy = -  np.sum(weights * residuals ** 2 / (sigmas ** 2 + y) ** 2)
            dy = sy / dsdy * (1 - sy / nu)
            y = y + dy
            sy = np.sum(weights * residuals ** 2 / (sigmas ** 2 + y))
        return np.sqrt(y)


def robust_dispersion_estimate(residuals):
    """Return half the intersextile range according to procedure P3 of LL83 (Eq. 22).

    Parameters
    ----------
    residuals : ndarray
        Array of residuals.

    Returns
    -------
    float
        The dispersion estimate.

    """
    quantiles = np.quantile(residuals, [1 / 6, 3 / 6, 5 / 6])
    return 0.5 * (quantiles[2] - quantiles[0]) + np.abs(quantiles[1])


def solve_linear_equations(design_matrix_coefficients: np.ndarray, dependent_variable: np.ndarray,
                           dependent_variable_error: np.ndarray, weights: np.ndarray = None, excess_noise: float = 0):
    """Use matrix inversion to solve the linear equations using least-squares.

    Note
    ----
    Weights and excess noise can be accounted for as described in procedure 4 of GAIA-C3-TN-LU-LL-083.

    Parameters
    ----------
    design_matrix_coefficients: ndarray
        The normal matrix.
    dependent_variable: ndarray
        The measurements.
    dependent_variable_error: ndarray
        The measurement uncertainties.
    weights: ndarray, optionl
        The measurement weights.
    excess_noise: float, optional
        The excess noise, a term that is added in quadrature to the measurement uncertainty.

    Returns
    -------
    ndarray
        The solution parameters.
    ndarray
        The residuals, i.e. observed minus calculated values.
    ndarray
        Inverse of the solution parameter covariance matrix.

    """
    D = design_matrix_coefficients
    h = dependent_variable

    if weights is None:
        weights = np.ones_like(dependent_variable)
    total_variance = dependent_variable_error ** 2 + excess_noise ** 2
    weight_parameters = weights / total_variance  # obsWeight in BasicSourceUpdateCalculator

    logging.debug(f'solve_linear_equations: \t excess_noise={excess_noise}')
    logging.debug(f'solve_linear_equations: \t weights={weights}')
    logging.debug(f'solve_linear_equations: \t total_variance={total_variance}')
    logging.debug(f'solve_linear_equations: \t weight_parameters (obsWeight)={weight_parameters}')
    WD = np.sqrt(weight_parameters).reshape(-1, 1) * D
    wh = np.sqrt(weight_parameters) * h
    parameter_covariance_matrix_formal_inverse = WD.T @ WD  # normalsFull in BasicSourceUpdateCalculator
    r = WD.T @ wh
    logging.debug(f'solve_linear_equations: \t r={r}')
    logging.debug(f'solve_linear_equations: \t parameter_covariance_matrix_formal_inverse '
                  f'(normalsFull)={parameter_covariance_matrix_formal_inverse}')
    parameters = np.linalg.solve(parameter_covariance_matrix_formal_inverse, r)
    residuals = dependent_variable - design_matrix_coefficients @ parameters

    return parameters, residuals, parameter_covariance_matrix_formal_inverse


def get_sextiles(x):
    """Return sextiles of input array.

    Note
    ----
    Emulate gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.getSextiles(double[])

    The sextiles are returned as an array of length 5 (say, double[] sext), with
    sext[0] = the 1st (lowest) sextile and sext[4] = the 5th (highest) sextile.
    sext[2] is the median, and (sext[4]-sext[0])/2 is half the intersextile
    range, which can be used as a robust estimate of the standard deviation.

    If the length of the input array x is <= 3, then sext[0] and sext[4] equal
    the minimum and maximum x values, respectively. If the length is 1, then all
    sextiles equal the single x value.

    Parameters
    ----------
    x : ndarray
        An array containg the input values.

    Returns
    -------
    ndarray
        Array containing the sextiles.

    """
    m = 6
    n = len(x)
    xs = copy.deepcopy(x)
    xs = np.sort(xs)
    sext = np.zeros(m - 1)
    for i in range(m - 1):
        p = (i + 1) / m
        lo = int(np.floor(p * n - 0.5))
        if lo < 0:
            sext[i] = xs[0]
        elif (lo > n - 2):
            sext[i] = xs[n - 1]
        else:
            h = p * n - (lo + 0.5)
            sext[i] = (1.0 - h) * xs[lo] + h * xs[lo + 1]
    return sext


@dataclass
class DesignEquation:
    """Class to handle and solve linear equation systems like AGIS does."""

    _required_parameters = ['design_matrix_coefficients', 'dependent_variable', 'dependent_variable_error']
    _default_array_parameters = {'excess_attitude_noise': 0., 'weight': 1., 'weight_prior': 1.}
    _default_parameters = {'n_outliers': 0, 'excess_source_variance': 0., 'n_solve': 0, 'gaussian_priors': None}

    def __init__(self, parameters: dict):
        """Initialise the object.

        Parameters
        ----------
        parameters: dict
            Dictionary of inputs for constructing the design equation. The dictionary  must have
            at least three entries as indicated by the `_required_parameters` attribute.

        """
        self.design_matrix_coefficients: np.ndarray
        self.dependent_variable: np.ndarray
        self.dependent_variable_error: np.ndarray
        self.excess_attitude_noise: np.ndarray
        self.weight: np.ndarray
        self.weight_prior: np.ndarray
        self.n_outliers: int
        self.excess_source_variance: float
        self.n_solve: int
        self.observationVariance: np.ndarray

        for key, item in parameters.items():
            setattr(self, key, item)
        for key in self._required_parameters:
            assert getattr(self, key) is not None
            # assert hasattr(self, key)

        for key, value in self._default_array_parameters.items():
            if hasattr(self, key) is False:
                setattr(self, key, np.ones(len(getattr(self, self._required_parameters[0]))) * value)
        for key, value in self._default_parameters.items():
            if hasattr(self, key) is False:
                setattr(self, key, value)

        # compute excess_attitude_noise (epsilonA) when necessary
        if hasattr(self, 'observationVariance'):
            if np.all(self.observationVariance != self.observation_variances):
                logging.debug("DesignEquation: excess_attitude_noise is not zero; "
                              "recomputing and setting it.")
                self.excess_attitude_noise = np.sqrt(self.observationVariance - self.dependent_variable_error ** 2)
                np.testing.assert_allclose(self.observationVariance, self.observation_variances)

        if hasattr(self, 'model') is False:
            self.model = 'unnamed_model'

    @property
    def observation_variances(self):
        """Return the observation variances.

        Note
        ----
        This is the sum of observation uncertainties and axcess attitude/observation noise,
        see Eq. 62ff of `Lindegren at al. 2012 <http://adsabs.harvard.edu/abs/2012A%26A...538A..78L>`__:

        sigmaTilde_l^2 = sigma_l^2 + epsilon_a^2(t_l),

        where obsSigma = the sigmas of the observations

        epsilonA = Any contribution from the excess attitude noise and excess calibration noise
        (but not the excess source noise).

        See Java method setObservationVariances(double[] obsSigma, double[] epsilonA)

        Returns
        -------
        ndarray
            The observation variances.

        """
        obs_variances = self.dependent_variable_error**2 + self.excess_attitude_noise**2
        return obs_variances

    @property
    def n_measurements(self):
        """Return current number of measurements."""
        return len(self.dependent_variable)

    @property
    def n_parameters(self):
        """Return current number of parameters."""
        return self.design_matrix_coefficients.shape[1]

    @property
    def n_degrees_of_freedom(self):
        """Return current degrees of freedom."""
        return self.n_measurements - self.n_parameters

    @property
    def n_effective_degrees_of_freedom(self):
        """Return current effective degrees of freedom."""
        return self.n_degrees_of_freedom - self.n_outliers

    @property
    def excess_source_noise(self):
        """Return current excess source noise."""
        return np.sqrt(self.excess_source_variance)

    def solve(self, solver='agis', **kwargs):
        """Solve the equations and return results.

        Parameters
        ----------
        solver : str
            the solver to be used, default is 'agis'
        kwargs : dict
            keyword arguments passed on to the solver code

        Returns
        -------
        dict
            Solution parameters and auxiliary information

        """
        if solver == 'least_squares':
            return self.solve_least_squares(**kwargs)
        elif solver == 'agis':
            return self.solve_like_agis(**kwargs)
        else:
            raise NotImplementedError

    def basic_source_update_calculator_solve(self, cholesky: bool = False, total_variance: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Use matrix inversion to solve the linear equations using least-squares.

        Note
        ----
        Modelled after gaia.cu3.agistools.algo.gis.source.BasicSourceUpdateCalculator.solve()

        See also solve_linear_equations function.

        Parameters
        ----------
        cholesky: boolean, optional
            Cholesky decomposition. Default: False.
        total_variance: narray, optional
            Total variance array. Default: None.

        Returns
        -------
        narray
            normals_full, which is the same as parameter_covariance_matrix_formal_inverse.
        narray
            rhs, right-hand side of the normal equations.
        narray
            Array of parameters.
        narray
            Array of residuals.

        """
        normals_full, rhs = self.construct_full_normals(total_variance=total_variance)

        if cholesky is False:
            parameters = np.linalg.solve(normals_full, rhs)
        else:
            logging.debug("basic_source_update_calculator_solve using cho_solve")
            logging.debug(f"basic_source_update_calculator_solve normals_full = {normals_full[0:5]}...")
            c, low = cho_factor(normals_full)
            logging.debug(f"basic_source_update_calculator_solve c = {c}")
            logging.debug(f"basic_source_update_calculator_solve low = {low}")
            parameters = cho_solve((c, low), rhs)

        logging.debug(f"basic_source_update_calculator_solve parameters = {parameters}")
        residuals = self.compute_residuals(parameters)

        # update counter
        self.n_solve += 1
        return normals_full, rhs, parameters, residuals

    def compute_residuals(self, parameters):
        """Compute the residuals given the input parameters.

        Parameters
        ----------
        parameters: narray
            Array of parameters corresponding to a solution of the system of equations.

        Returns
        -------
        narray
            The computed residuals.

        """
        return self.dependent_variable - self.design_matrix_coefficients @ parameters

    def get_postfit_residuals(self):
        """Return post-fit residuals.

        Note
        ----
        See Java method gaia.cu3.agistools.algo.gis.source.BasicSourceUpdateCalculator.getPostfitResiduals()

        """
        return self.basic_source_update_calculator_solve()[-1]

    def get_updates(self):
        """Return the updates from the latest call to solve().

        Note
        ----
        See Java method gaia.cu3.agistools.algo.gis.source.BasicSourceUpdateCalculator.getUpdates()

        """
        return self.basic_source_update_calculator_solve()[-2]

    def get_prefit_residuals(self):
        """Return pre-fit residuals.

        Note
        ----
        See Java method gaia.cu3.agistools.algo.gis.source.BasicSourceUpdateCalculator.getPrefitResiduals()

        """
        return self.dependent_variable

    @staticmethod
    def get_initial_dispersion_estimate(data):
        """Return initial dispersion estimate of the input dataset.

        Note
        ----
        See gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.getInitialDispersionEstimate(double[])

        Parameters
        ----------
        data : ndarray
            Input dataset.

        Returns
        -------
        ndarray
            Initial dispersion estimate of the input.

        """
        quantile_values = np.quantile(data, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        return 0.5 * (quantile_values[4] - quantile_values[0]) + np.abs(quantile_values[2])

    def compute_n_outliers(self):
        """Compute the number of outliers based on weights and threshold.

        Note
        ----
        See gaia.cu3.agistools.algo.gis.source.SourceUpdateStatistics.getNOut()

        Returns
        -------
        narray
            n_outliers

        """
        downweights = self.get_downweights(True)  # return downweight * downweight_prior
        n_outliers = np.sum(downweights < AGIS_SOLVER_OPTIONS['downweightLimit'])
        return n_outliers

    def get_index_keep(self):
        """Return the indices of the measurements that are not outliers."""
        downweights = self.get_downweights(True)
        index_keep = np.where(downweights >= AGIS_SOLVER_OPTIONS['downweightLimit'])[0]
        if (len(index_keep) + self.compute_n_outliers() != len(self.weight)):
            raise ValueError("Number of outliers is inconsistent.")
        return index_keep

    def calculate_estimate(self):
        """Calculate the solution estimate.

        Note
        ----
        See gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.calculateEstimate()

        Returns
        -------
        narray
            Excess source noise.
        narray
            Excess source noise significance.

        """
        # try first with excess source variance = 0
        excess_source_noise = 0
        self.excess_source_variance = excess_source_noise**2

        # gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.setDefaultDownweights()
        self.weight = np.ones(self.n_measurements)

        iterRobust = 0
        ssr, ssr_derivative = self.get_weighted_ssr_and_derivative()
        sMin0 = ssr

        dof = self.n_effective_degrees_of_freedom
        logging.debug(f"initial dof = {dof}")
        # iter_excess = 0
        excess_source_noise_significance = 0.

        # if sMin0 <= dof we are done; otherwise try to set a reasonable excess source
        # noise to eliminate gross outliers
        if (sMin0 > dof):
            excess_source_variance_old = self.excess_source_variance
            excess_source_noise_improved = self.get_initial_dispersion_estimate(self.get_prefit_residuals())
            logging.debug(f"self.get_initial_dispersion_estimate(self.get_prefit_residuals()) = \
            {self.get_initial_dispersion_estimate(self.get_prefit_residuals())}")
            excess_source_variance_improved = excess_source_noise_improved**2
            self.excess_source_variance = excess_source_variance_improved
            logging.debug(f"self.excess_source_variance = {self.excess_source_variance}")

            logging.debug(f"calculate_estimate self.get_prefit_residuals() = {self.get_prefit_residuals()[0:4]}...")
            logging.debug(f"self.observation_variances = {self.observation_variances[0:4]}")
            downweights = self.calculate_downweights(self.get_prefit_residuals())
            logging.debug(f"downweights!=1 = {downweights[np.where(downweights != 1)[0]]}")
            self.weight = downweights
            self.n_outliers = self.compute_n_outliers()
            dof = self.n_effective_degrees_of_freedom
            logging.debug(f"updated dof = {dof}")

            excess_source_variance_delta = excess_source_variance_improved - excess_source_variance_old
            logging.debug(f"Before robust loop: excess_source_variance_delta = \
            {excess_source_variance_delta}, excess_source_variance={self.excess_source_variance}")
            while (np.abs(excess_source_variance_delta) > AGIS_SOLVER_OPTIONS['tolEsvRobust'] * self.excess_source_variance) \
                    and (iterRobust < AGIS_SOLVER_OPTIONS['maxIterRobust']):
                logging.debug(f"Robust estimator iteration {iterRobust}: excess_source_variance={self.excess_source_variance}")
                iterRobust += 1
                excess_source_variance_old = self.excess_source_variance
                excess_source_variance_improved = self.get_improved_estimate_of_excess_source_variance(dof)
                logging.debug(f"excess_source_variance_improved = {excess_source_variance_improved}")
                logging.debug(f"Robust estimator iteration {iterRobust}: \
                self.get_improved_estimate_of_excess_source_variance(dof) = {self.get_improved_estimate_of_excess_source_variance(dof)}")

                self.excess_source_variance = excess_source_variance_improved
                self.weight = self.calculate_downweights(self.get_postfit_residuals())
                self.n_outliers = self.compute_n_outliers()
                dof = self.n_effective_degrees_of_freedom
                logging.debug(f"iterRobust={iterRobust}: dof = {dof}")

                excess_source_variance_delta = excess_source_variance_improved - excess_source_variance_old
                logging.debug(f"Robust estimator iteration {iterRobust}: excess_source_variance = \
                {self.excess_source_variance}; excess_source_variance_delta = {excess_source_variance_delta}")

            excess_source_noise, excess_source_noise_significance = self.calculate_excess_source_noise()

        return excess_source_noise, excess_source_noise_significance

    def solve_like_agis(self, total_variance: np.ndarray = None) -> dict:
        """Solve linear equations like AGIS.

        Parameters
        ----------
        total_variance: float
            Total variance.

        Returns
        -------
        dict
            Dictionary with information on the solution.

        """
        if total_variance is None:
            excess_source_noise, excess_source_noise_significance = self.calculate_estimate()
            logging.debug(f"solve_like_agis: excess_source_noise = {excess_source_noise}")
        else:
            logging.debug('Computing direct solution using fixed weights and external total variance (no inner iterations).')
            excess_source_noise = None
            excess_source_noise_significance = None

        # get the best-fit parameters of the model
        parameter_covariance_matrix_formal_inverse, rhs, parameters, residuals = self.basic_source_update_calculator_solve(total_variance=total_variance)

        results = OrderedDict()
        results['model'] = self.model
        results['solver'] = 'agis'
        results['parameters'] = parameters

        # keep  only ‘good’ (i.e. not strongly downweighted) observations for fit metric calculations
        # this is consistent with documentation:
        # https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
        index_keep = self.get_index_keep()
        results['residuals'] = residuals[index_keep]

        results['index_keep'] = index_keep

        #  optional: specify timestamps of used observations
        if hasattr(self, 'timestamps'):
            results['timestamps_of_used_observations'] = self.timestamps.reset_index(drop=True).iloc[index_keep]

        results['weights'] = self.weight[index_keep]
        results['n_data_total'] = self.n_measurements

        # number of measurements/constraints, i.e. number of equations
        results['n_measurements'] = self.n_measurements - self.compute_n_outliers()

        results['parameter_covariance_matrix_formal_inverse'] = parameter_covariance_matrix_formal_inverse
        results['parameter_covariance_matrix_formal'] = np.linalg.inv(results['parameter_covariance_matrix_formal_inverse'])

        # formal uncertainty  of the solution parameters
        results['parameters_formal_uncertainty'] = np.sqrt(np.diag(results['parameter_covariance_matrix_formal']))

        # number free parameters
        results['n_parameters'] = len(results['parameters'])

        results['excess_noise'] = excess_source_noise
        results['n_outliers'] = self.compute_n_outliers()
        results['significance'] = excess_source_noise_significance

        # check setting of total variance, since this goes into ln likelyhood and BIC
        # total variance of measurements
        # results['total_variance'] = self.dependent_variable_error[index_keep]**2 + excess_source_noise**2
        if total_variance is None:
            results['total_variance'] = self.observation_variances[index_keep] + excess_source_noise**2
        else:
            results['total_variance'] = total_variance

        # results['measurement_variance'] = self.dependent_variable_error[index_keep]**2
        results['measurement_variance'] = self.observation_variances[index_keep]

        # parameter covariance matrix normalised to yield chi2=1
        chi2 = chi_squared(results['residuals'], results['measurement_variance'])
        results['parameter_covariance_matrix_normalised'] = results['parameter_covariance_matrix_formal'] * chi2 / self.n_effective_degrees_of_freedom
        # normalised uncertainty of the solution parameters
        results['parameters_normalised_uncertainty'] = np.sqrt(np.diag(results['parameter_covariance_matrix_normalised']))

        keys_for_stats = SolutionStatistic._required_parameters + ['model', 'solver', 'excess_noise', 'n_outliers']
        results['solution_statistic'] = SolutionStatistic({key: results[key] for key in keys_for_stats})

        logging.debug('solve_like_agis result: {}/{} measurements are considered outliers'.format(results['n_outliers'], results['n_data_total']))

        return results

    def solve_least_squares(self):
        """Solve equations with a standard least-square solver.

        Returns
        -------
        dict
            Dictionary containing information about the solution.

        """
        # solve the equations
        parameters, residuals, N = solve_linear_equations(self.design_matrix_coefficients,
                                                          self.dependent_variable, self.dependent_variable_error)

        results = OrderedDict()
        results['solver'] = 'least-squares'
        results['model'] = self.model
        results['parameters'] = parameters
        results['residuals'] = residuals
        results['parameter_covariance_matrix_formal_inverse'] = N
        results['parameter_covariance_matrix_formal'] = np.linalg.inv(results['parameter_covariance_matrix_formal_inverse'])

        # formal uncertainty of the solution parameters
        results['parameters_formal_uncertainty'] = np.sqrt(np.diag(results['parameter_covariance_matrix_formal']))

        # number of measurements/constraints, i.e. number of equations
        results['n_measurements'] = len(self.dependent_variable)

        # number of free parameters
        results['n_parameters'] = len(results['parameters'])

        # total variance of measurements
        results['total_variance'] = self.dependent_variable_error ** 2
        results['measurement_variance'] = results['total_variance']

        # parameter covariance matrix normalised to yield chi2=1
        chi2 = chi_squared(results['residuals'], results['measurement_variance'])
        results['parameter_covariance_matrix_normalised'] = results['parameter_covariance_matrix_formal'] * chi2 / self.n_effective_degrees_of_freedom
        # normalised uncertainty of the solution parameters
        results['parameters_normalised_uncertainty'] = np.sqrt(np.diag(results['parameter_covariance_matrix_normalised']))

        keys_for_stats = SolutionStatistic._required_parameters + ['model', 'solver']
        results['solution_statistic'] = SolutionStatistic({key: results[key] for key in keys_for_stats})

        return results

    def construct_full_normals(self, total_variance=None):
        """Replicate Java method constructFullNormals.

        Note
        ----
        gaia.cu3.agistools.algo.gis.source.BasicSourceUpdateCalculator.constructFullNormals()

        Parameters
        ----------
        total_variance: narray, optional
            Array of total variances.

        Returns
        -------
        narray
            Parameter covariance matrix formal inverse.
        narray
            rhs, right-hand side of the normal equations.

        """
        weights = self.weight * self.weight_prior

        logging.debug(f"construct_full_normals: \t self.excess_source_variance = {self.excess_source_variance}")
        if total_variance is None:
            total_variance = self.observation_variances + self.excess_source_variance

        weight_parameters = weights / total_variance  # obsWeight in BasicSourceUpdateCalculator
        logging.debug(f'construct_full_normals: \t weight_parameters (obsWeight) = {weight_parameters[0:4]}...')
        logging.debug(f'construct_full_normals: \t design_matrix_coefficients = {self.design_matrix_coefficients}...')

        WD = np.sqrt(weight_parameters).reshape(-1, 1) * self.design_matrix_coefficients  # == Math.sqrt(obsWeight) * obs.partialDerivatives
        wh = np.sqrt(weight_parameters) * self.dependent_variable  # == Math.sqrt(obsWeight) * obs.prefitResidual

        logging.debug(f'construct_full_normals: \t Math.sqrt(obsWeight) * obs.partialDerivatives=(normalsLhs)=(WD)={WD[0:4]}...')
        logging.debug(f'construct_full_normals: \t Math.sqrt(obsWeight) * obs.prefitResidual=(wh)={wh[0:4]}...')

        rhs = WD.T @ wh  # right-hand side of the normal equations

        parameter_covariance_matrix_formal_inverse = WD.T @ WD  # normalsFull in BasicSourceUpdateCalculator

        # add a prior when specified
        if self.gaussian_priors is not None:
            logging.debug(self.gaussian_priors)
            # add Gaussian prior on the matrix diagonal
            diagonal_indices = np.diag_indices_from(parameter_covariance_matrix_formal_inverse)
            for ii, prior in enumerate(self.gaussian_priors):
                if prior is not None:
                    parameter_covariance_matrix_formal_inverse[diagonal_indices[0][ii], diagonal_indices[1][ii]] += 1 / (float(prior) ** 2)

        logging.debug(f'construct_full_normals: \t parameter_covariance_matrix_formal_inverse (normalsFull) = \
        {parameter_covariance_matrix_formal_inverse}')
        logging.debug(f'construct_full_normals: \t rhs (rhs)={rhs}')

        return parameter_covariance_matrix_formal_inverse, rhs

    def get_weighted_ssr_and_derivative(self):  # , excess_source_variance=0):
        """Return weighted sum of squared residuals and its derivative."""
        normals_full, rhs = self.construct_full_normals()
        parameters = np.linalg.solve(normals_full, rhs)
        residuals = self.dependent_variable - self.design_matrix_coefficients @ parameters
        logging.debug(f"get_weighted_ssr_and_derivative PostfitResiduals = {residuals[0:4]}...")

        ssr = np.sum(residuals ** 2 * self.weight * 1 / (self.observation_variances + self.excess_source_variance))
        ssr_derivative = -1 * np.sum(residuals ** 2 * self.weight * (1 / (self.observation_variances + self.excess_source_variance)) ** 2)
        return ssr, ssr_derivative

    def get_improved_estimate_of_excess_source_variance(self, dof):
        """Return an improved estimate of the excess source noise.

        Note
        ----
        This uses the currently set data, variances and downweights.
        It implements a single iteration of algorithm P1 in GAIA-C3-TN-LU-LL-083.

        See the Java method getImprovedEstimateOfExcessSourceVariance(int dof).

        Parameters
        ----------
        dof : int
            degrees of freedom

        Returns
        -------
        float
            Applied change in excess source variance.

        """
        s = self.get_weighted_ssr_and_derivative()
        logging.debug(f"get_improved_estimate_of_excess_source_variance  self.get_weighted_ssr_and_derivative() \
        = {self.get_weighted_ssr_and_derivative()}")
        deltaVar = -(s[0] / s[1]) * (s[0] / dof - 1.0)
        return np.max(np.array([self.excess_source_variance + deltaVar, 0.0]))

    def calculate_excess_source_noise(self):
        """Compute and return excess source noise.

        Note
        ----
        Calculates and sets the excess source noise and significance for the currently set (fixed)
        variances and downweights. This implements algorithm P1 in GAIA-C3-TN-LU-LL-083.

        See corresponding Java code:
        gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.calculateExcessSourceNoise()

        Returns
        -------
        float
            The source excess noise.
        float
            The source excess noise significance.

        """
        self.excess_source_variance = 0
        excess_source_noise_significance = 0.

        ssr, ssr_derivative = self.get_weighted_ssr_and_derivative()
        sMin0 = ssr

        dof = self.n_effective_degrees_of_freedom
        logging.debug(f"calculate_excess_source_noise: dof = {dof}")
        iter_excess = 0
        if (sMin0 > dof):
            excess_source_variance_old = self.excess_source_variance
            excess_source_variance_improved = self.get_improved_estimate_of_excess_source_variance(dof)
            self.excess_source_variance = excess_source_variance_improved
            excess_source_variance_delta = excess_source_variance_improved - excess_source_variance_old
            while (np.abs(excess_source_variance_delta) > AGIS_SOLVER_OPTIONS['tolEsvExcess'] * self.excess_source_variance) \
                    and (iter_excess < AGIS_SOLVER_OPTIONS['maxIterExcess']):
                logging.debug(f"calculate_excess_source_noise: Excess noise iteration {iter_excess}: \
                excess_source_variance = {self.excess_source_variance}")
                iter_excess += 1
                excess_source_variance_old = self.excess_source_variance
                excess_source_variance_improved = self.get_improved_estimate_of_excess_source_variance(dof)
                self.excess_source_variance = excess_source_variance_improved
                excess_source_variance_delta = excess_source_variance_improved - excess_source_variance_old
                logging.debug(f"calculate_excess_source_noise: Excess noise iteration {iter_excess}: \
                excess_source_variance = {self.excess_source_variance}")

            # // Eq.(10)
            excess_source_noise_significance = (sMin0 - dof) / np.sqrt(2.0 * dof)

        return self.excess_source_noise, excess_source_noise_significance

    def get_downweights(self, multiply_by_prior: bool):
        """Return array of the currently defined downweights.

        Returns
        -------
        ndarray
            Array of downweights.

        """
        downweights = self.weight
        if multiply_by_prior:
            downweights *= self.weight_prior
        return downweights

    def calculate_downweights(self, residuals):
        """Return downweights for a given array of residuals.

        Note
        ----
        See gaia.cu3.agistools.algo.gis.source.RobustSourceUpdateCalculator.calculateDownweights.
        Calculates downweights for a given array of residuals, using the current variances and
        excess source noise. This implements algorithm P2, i.e. Eq. (20) in GAIA-C3-TN-LU-LL-083.

        Parameters
        ----------
        residuals : ndarray
            Array of residuals.

        Returns
        -------
        ndarray
            Array of downweights.

        """
        downweights = self.get_downweights(False)
        total_variance = self.observation_variances + self.excess_source_variance
        normalised_residuals = residuals / np.sqrt(total_variance)
        downweights_new = AGIS_SOLVER_OPTIONS['downweighter'](normalised_residuals, AGIS_SOLVER_OPTIONS['dfWeight'])

        # Only update the downweights if they are being used
        index_used = np.where(self.weight_prior == 1)[0]
        downweights[index_used] = downweights_new[index_used]

        return downweights


def agis_weight_function(z):
    """Evaluate AGIS weight function as defined in Eq. 66 of Lindegren et al. 2012.

    Parameters
    ----------
    z : ndarray
        Input data array

    Returns
    -------
    ndarray
        An array on weights

    """
    x = np.abs(z)
    return np.piecewise(x,
                        [x <= 2, (2 < x) & (x < 3), x >= 3],
                        [1, lambda x: 1 - 1.773735 * (x - 2) ** 2 + 1.141615 * (x - 2) ** 3,
                            lambda x: np.exp(-x / 3)])


def agis_weights(residuals, uncertainties, excess_noise, weight_threshold=0.2):
    """Return the residual down weight factors and identify outliers.

    Note
    ----
    See procedure P2 in GAIA-C3-TN-LU-LL-083 (Eq. 21) from the `Archive documentation <https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html>`__.

    astrometric_n_good_obs_al : Number of good observations AL (short)
    Number of AL observations (= CCD transits) that were not strongly downweighted in the astrometric
    solution of the source. Strongly downweighted observations (with downweighting factor 𝑤<0.2)
    are instead counted in astrometric_n_bad_obs_al.

    The sum of astrometric_n_good_obs_al and astrometric_n_bad_obs_al equals astrometric_n_obs_al,
    the total number of AL observations used in the astrometric solution of the source.

    Parameters
    ----------
    residuals: narray
        Array of residuals.
    uncertainties: narray
        Array of uncertainties.
    excess_noise: narray.
        Array of excess noises.
    weight_threshold: float
        Cutoff below which measurements are considered to be outliers.

    Returns
    -------
    ndarray
        Array of weights.
    int
        Number of outliers.
    ndarray
        Array of the good indices.

    """
    if np.any(np.sqrt(uncertainties ** 2 + excess_noise ** 2) == 0):
        logging.debug(f'excess_noise={excess_noise}')
        logging.warning('agis_weights was given uncertainties with at least one zero entry.')

    argument = residuals / np.sqrt(uncertainties ** 2 + excess_noise ** 2)  # normalizedResidual in calculateDownweights

    # compute weights
    weights = agis_weight_function(argument)

    # outlier_index = np.where(weights < weight_threshold)[0]

    # index of ‘good’ (i.e. not strongly downweighted) observations
    good_index = np.where(weights >= weight_threshold)[0]

    # compute number of identified outliers
    n_outliers = len(residuals) - len(good_index)
    return weights, n_outliers, good_index
