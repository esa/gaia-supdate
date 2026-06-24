#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Module providing statistical metrics and an auxiliary class."""

__author__ = "Johannes Sahlmann"

import logging

import numpy as np
from scipy.special import betainc as betai


class SolutionStatistic:
    """Convenience class for handling solutions of linear equations systems."""

    _required_parameters = ['n_parameters', 'n_measurements', 'residuals', 'total_variance',
                            'measurement_variance']

    def __init__(self, parameters):
        """Initialise.

        Parameters
        ----------
        parameters : dict

        """
        for key, item in parameters.items():
            setattr(self, key, item)
        for key in self._required_parameters:
            assert hasattr(self, key)

        if hasattr(self, 'model') is False:
            self.model = f'na_{self.n_parameters}p'
        if hasattr(self, 'solver') is False:
            self.solver = 'na'
        if hasattr(self, 'excess_noise') is False:
            self.excess_noise = None
        if hasattr(self, 'n_outliers') is False:
            self.n_outliers = 0

        assert self.n_measurements == len(self.residuals)

        # TODO
        # define these as properties

        # number of degrees of freedom
        self.n_degrees_of_freedom = self.n_measurements - self.n_parameters

        # chi squared
        # from MDB/CU3/AGIS/Source datamodel: These $\chi^2$ values were computed for all accepted
        # observations of the source, without taking into account the source $\tt excessNoise$ (if any).
        # measurement_variance = formal uncertainty + attitude excess noise
        self.chi2 = chi_squared(self.residuals, self.measurement_variance)
        self.chi2_total = chi_squared(self.residuals, self.total_variance)

        # log of the likelihood
        self.ln_likelihood = log_likelihood(self.residuals, self.total_variance)

        # Bayesian Information Criterion (BIC), see e.g. https://en.wikipedia.org/wiki/Bayesian_information_criterion
        self.bic = bayesian_information_criterion(self.n_parameters, self.n_measurements,
                                                  self.ln_likelihood)
        self.aic = akaike_information_criterion(self.n_parameters, self.ln_likelihood)

        # Gaia F2
        self.f2 = gaia_f2(self.chi2, self.n_degrees_of_freedom)
        # compute also when using total variance in chi2
        self.f2_total = gaia_f2(self.chi2_total, self.n_degrees_of_freedom)

        # RMS dispersion of the residuals
        self.residuals_rms = np.std(self.residuals)
        self.residuals_mean = np.mean(self.residuals)

    def __add__(self, other):
        """Compute the fit statistics for a pair of independent results, e.g. 2 x 5p."""
        combined_parameters = {}
        for key in ['n_parameters', 'n_measurements', 'n_outliers']:
            combined_parameters[key] = getattr(self, key) + getattr(other, key)
        for key in ['residuals', 'total_variance', 'measurement_variance']:
            combined_parameters[key] = np.concatenate((getattr(self, key), getattr(other, key)))
        for key in ['excess_noise']:
            combined_parameters[key] = np.array([getattr(self, key), getattr(other, key)])
        return SolutionStatistic(combined_parameters)

    def __gt__(self, other):
        """Decide whether self or other is a statistically better solution.

        For two solutions s1 and s2, s1 > s2 if s1 is the "better solution".

        Several metrics are computed (deltaBIC, F-test, ...), but only one score metric will be
        defined as the decisive one.

        """
        # F-TEST (assumes everything is perfectly Gaussian.)
        # number of data points has to be equal
        if self.n_measurements != other.n_measurements:
            return None

        # probability that the simpler model is better
        f_test_probability = f_test_of_additional_parameter(self.n_measurements, self.n_parameters,
                                                            self.chi2, other.n_parameters,
                                                            other.chi2)
        print(f'F-test prob: {f_test_probability:.1e}')
        f_test_probability_threshold = 1e-4

        if (f_test_probability is None) or (f_test_probability < f_test_probability_threshold):
            return False
        else:
            return True

    def __str__(self):
        """Return string describing the instance."""
        string_0 = f'model={self.model} solver={self.solver}\nn_meas={self.n_measurements} \
        n_param={self.n_parameters} n_free={self.n_degrees_of_freedom} n_outliers={self.n_outliers}'
        string_1 = f'omc_rms={self.residuals_rms:.3f} f2={self.f2:.3f} bic={self.bic:.3f} chi2={self.chi2:.3f}'
        if self.excess_noise is not None:
            if np.ndim(self.excess_noise) == 0:
                string_1 += f'\nexcess_noise={self.excess_noise:.3e}'
            else:
                string_1 += f"\nexcess_noise={['{:.3e}'.format(x) for x in self.excess_noise if x is not None]}"
        return f'SolutionStatistic: {string_0}\n{string_1}'


def chi_squared(residuals, variance):
    """Return chi2 fit quality metric.

    Parameters
    ----------
    residuals: narray
        Array of residuals.
    variance: narray
        Array of variances.

    Returns
    -------
    float
        chi2 fit quality metric

    """
    return np.sum(residuals ** 2 / variance)


def bayesian_information_criterion(n_parameters, n_measurements, ln_likelihood):
    """Return BIC.

    Parameters
    ----------
    n_parameters: int
        Number of parameters.
    n_measurements: int
        Number of measurements.
    ln_likelihood: float
        Logarithm of the likelihood

    Returns
    -------
    float
        Bayesian information criterion

    """
    bic = n_parameters * np.log(n_measurements) - 2 * ln_likelihood
    return bic


def akaike_information_criterion(n_parameters, ln_likelihood):
    """Return AIC.

    Parameters
    ----------
    n_parameters: int
        Number of parameters.
    n_measurements: int
        Number of measurements.

    Returns
    -------
    float
        Akaike information criterion

    """
    aic = 2 * n_parameters - 2 * ln_likelihood
    return aic


def log_likelihood_0(variance):
    """Return lnL0 term.

    Parameters
    ----------
    variance: narray
        Array of variances.

    Returns
    -------
    float
        lnL0

    """
    lnL0 = np.sum(np.log(np.sqrt(variance * 2 * np.pi)))
    return lnL0


def log_likelihood(residuals, variance):
    """Return the log(likelihood).

    Parameters
    ----------
    residuals: narray
        Array of residuals.
    variance: narray
        Array of variances.

    Returns
    -------
    float
        ln_likelihood

    """
    chi2 = chi_squared(residuals, variance)
    lnL0 = log_likelihood_0(variance)
    ln_likelihood = -0.5 * (chi2) - lnL0
    return ln_likelihood


def f_test_of_additional_parameter(n_measurements: int, n_parameters_1: int, chi2_1: float,
                                   n_parameters_2: int, chi2_2: float) -> float:
    """Return F-Test probability that the simpler model is correct.

    Note
    ----
    e.g. n_parameters_1 = 5.  is the number of PPM parameters

    e.g. n_parameters_2 = n_parameters_1 + 7. is the number of PPM + orbital parameters

    Parameters
    ----------
    n_measurements: int
        Number of data points
    n_parameters_1: int
        Number of parameters of the simpler model
    chi2_1: float
        chi^2 corresponding to the simpler model
    n_parameters_2: int
        Number of parameters of the model with more parameters
        n_parameters_2 > p1
    chi2_2: float
        chi^2 corresponding to the model with more parameters

    Returns
    -------
    float
        Probability.

    """
    nu1 = n_parameters_2 - n_parameters_1
    nu2 = n_measurements - n_parameters_2  # degrees of freedom

    # the self.model has to be simpler than the other.model
    if np.any(n_parameters_1 >= n_parameters_2):
        raise ValueError('First model has to be simpler than second model.')

    if np.any(chi2_1 < chi2_2):
        logging.warning('Solution better with less parameters.')
        return 0.

    # F test
    F0 = nu2 / nu1 * (chi2_1 - chi2_2) / chi2_2

    # probability
    prob = betai(0.5 * nu2, 0.5 * nu1, nu2 / (nu2 + F0 * nu1))

    return prob


def gaia_f2(chi2, nu):
    """Return F2, a goodness-of-fit statistic.

    Note
    ----
    In the Gaia archive this is called astrometric_gof_al, see the `Archive documentation <https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html>`__.

    Parameters
    ----------
    chi2 : float
        goodness-of-fit statistic chi^2.
    nu : int
        number of degrees of freedom.

    Returns
    -------
    float
        Gaia f2, which is a goodness of fit statistic.

    """
    return np.sqrt(9 * nu / 2) * ((chi2 / nu) ** (1 / 3) + 2 / (9 * nu) - 1)
