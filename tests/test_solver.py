#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the utils.agissolvers module."""

__author__ = "Johannes Sahlmann"

from collections import OrderedDict
import logging
import os

import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.solver import DesignEquation, decay_downweight, huber_downweight

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rad2mas = u.rad.to(u.milliarcsecond)


def get_test_data(data_file: str, filter: str = 'None'):
    """Return dataframe with test data.

    Parameters
    ----------
    coefficients_file : file

    Returns
    -------
    df : dataframe
    design_parameters : dictionary

    """
    logging.info(f'Using input file {data_file}')
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'Initial number of rows {len(df)}')

    if filter == 'Al':
        df = df[df['fprsDirectionIndex'] == 0]
        logging.info(f'Selected number of rows {len(df)} in {filter}')

    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', None)

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    return df, design_parameters


def test_against_SourceUpdateCalculatorIntegrationTest():
    """Test against the test solution obtained as part of the AGIS integration test.

    The gaia.cu3.agis.progs.test.SourceUpdateCalculatorIntegrationTest produces a solution and we
    chose one source from there. The csv file was exported for that source and we test the
    python implementation against the AGIS code.

    """
    # loop over two sources that have 5P solutions without priors
    for source_id in [1, 2]:
        # read data exported as part of the (modified) integration test, select only AL measuremnts

        coefficients_file = f'{source_id}_updtCalculator_coefficients.csv'
        data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)

        df, design_parameters = get_test_data(data_file, filter='Al')
        design_parameters['weight'] = df['downweight'].values
        design_parameters['weight_prior'] = df['downweightPrior'].values
        design_parameters['observationVariance'] = df['observationVariance'].values
        design_equation = DesignEquation(design_parameters)

        # solve the equations in an AGIS-like manner
        results = design_equation.solve_like_agis()

        # compare with test solution from integration test
        # read source parameters of the secondary update solution
        agis_test_source_parameters = pd.read_csv(os.path.join(TEST_DATA_ROOT, 'agis', f'{source_id}_agis_test_source_parameters.csv'))

        # read source solution parameters of the '0' solution in the store, the starting point for the secondary update
        # The pre-fit residuals that are used for the right-hand side of the design equation account for the input source parameters.
        # Therefore the fitted parameters are actually updates/corrections relative to the '0' solution
        agis_input_source_parameters = pd.read_csv(os.path.join(TEST_DATA_ROOT, 'agis', f'{source_id}_agis_input_source_parameters.csv'))

        # compare excess source noise results
        agis_source_column_mapping = {'excessNoise': 'excess_noise', 'excessNoiseSig': 'significance'}
        test_solution = agis_test_source_parameters.query("sourceId == @source_id")
        input_solution = agis_input_source_parameters.query("sourceId == @source_id")
        for key, mapped_key in agis_source_column_mapping.items():
            logging.info(results[mapped_key])
            assert_allclose(test_solution[key], results[mapped_key])

        # compare astrometric parameters
        corrected_parameter_fields = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta']
        astrometric_parameter_values = input_solution[corrected_parameter_fields].values
        astrometric_parameter_corrections = results['parameters'][0:5]

        astrometric_parameter_corrected_values = astrometric_parameter_values + astrometric_parameter_corrections
        assert_allclose(test_solution[corrected_parameter_fields].values, astrometric_parameter_values + astrometric_parameter_corrections)

        # compare their uncertainties
        corrected_parameter_error_fields = [f"{s}Error" for s in corrected_parameter_fields]

        # take care of naming conventions
        corrected_parameter_error_fields[corrected_parameter_error_fields.index('alphaError')] = 'alphaStarError'
        astrometric_parameter_corrections_uncertainties = results['parameters_formal_uncertainty'][0:5]

        # take care of unit conventions, convert from rad to mas
        astrometric_parameter_corrections_uncertainties[0:2] *= rad2mas

        expected_parameters = (test_solution[corrected_parameter_fields].values)[0]
        logging.info(f"expected_parameters {expected_parameters}")
        actual_parameters = astrometric_parameter_corrected_values[0]
        logging.info(f"actual_parameters   {actual_parameters}")
        logging.info(f"Input               {astrometric_parameter_values[0]}")
        logging.info(f"Corrections         {astrometric_parameter_corrections}")

        assert_allclose(test_solution[corrected_parameter_error_fields].values[0], astrometric_parameter_corrections_uncertainties)

        # compare chi2 and other metrics and number of rejected outliers
        assert_allclose(test_solution['chi2Al'], getattr(results['solution_statistic'], 'chi2'))
        assert_allclose(test_solution['f2'], getattr(results['solution_statistic'], 'f2'))

        logging.info(f"{source_id}: nObsAl={test_solution['nObsAl'].values[0]}; nOutliersAl={test_solution['nOutliersAl'].values[0]}")
        assert_allclose(test_solution['nObsAl'].values[0], getattr(results['solution_statistic'], 'n_measurements'))
        assert_allclose(test_solution['nOutliersAl'].values[0], getattr(results['solution_statistic'], 'n_outliers'))


def test_against_ConstrainedColourSourceUpdateCalculatorIntegrationTest():
    """Test against the test solution obtained as part of the AGIS integration test.

    The gaia.cu3.agis.progs.test.ConstrainedColourSourceUpdateCalculatorIntegrationTest produces a solution and we
    chose one source from there. The csv file was exported for that source and we test the
    python implementation against the AGIS code.

    """
    for source_id in [3]:
        # read data exported as part of the (modified) integration test, select only AL measuremnts
        coefficients_file = f'{source_id}_coefficients.csv'
        data_file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', coefficients_file)

        df, design_parameters = get_test_data(data_file, filter='Al')
        design_parameters['weight'] = df['downweight'].values
        design_parameters['weight_prior'] = df['downweightPrior'].values
        design_parameters['observationVariance'] = df['observationVariance'].values

        # define prior for constrained-colour update
        n_linear_parameters = design_parameters['design_matrix_coefficients'].shape[1]
        design_parameters['gaussian_priors'] = np.array([None] * (n_linear_parameters - 1) + [0.085e-3])
        design_parameters['model'] = 'constrained_colour'

        design_equation = DesignEquation(design_parameters)

        # read source solution parameters of the '0' solution in the store, the starting point for the secondary update
        # The pre-fit residuals that are used for the right-hand side of the design equation account for the input source parameters.
        # Therefore the fitted parameters are actually updates/corrections relative to the '0' solution
        agis_input_source_parameters = pd.read_csv(os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{source_id}_0_agis_input_source_parameters.csv'))

        # compare with test solution from integration test
        # read source parameters of the secondary update solution
        # agis_test_source_parameters = pd.read_csv(os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{source_id}_agis_test_source_parameters.csv'))
        TEST_SOLUTION_ID = 'X'
        agis_test_source_parameters = pd.read_csv(os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{source_id}_{TEST_SOLUTION_ID}_agis_test_source_parameters.csv'))

        # The solution was obtained with use_heaviside_weights = True
        excess_source_noise_mas = agis_test_source_parameters.loc[0, 'excessNoise']
        excess_source_variance = excess_source_noise_mas**2
        total_variance = design_equation.observation_variances + excess_source_variance
        logging.info(f"excess_source_noise_mas = {excess_source_noise_mas}")

        # solve the equations in an AGIS-like manner
        results = design_equation.solve_like_agis(total_variance=total_variance)

        # compare results
        test_solution = agis_test_source_parameters.query("sourceId == @source_id")
        input_solution = agis_input_source_parameters.query("sourceId == @source_id")

        # compare astrometric parameters
        if n_linear_parameters == 6:
            corrected_parameter_fields = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta', 'pseudoColor']
        else:
            corrected_parameter_fields = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta']
        astrometric_parameter_values = (input_solution[corrected_parameter_fields].values)[0]
        astrometric_parameter_corrections = results['parameters']

        astrometric_parameter_corrected_values = astrometric_parameter_values + astrometric_parameter_corrections

        # pseudoColor update has not been added to the colour parameter by AGIS, hence we compare only the fitted offset
        if n_linear_parameters == 6:
            astrometric_parameter_corrected_values[-1] = astrometric_parameter_corrections[-1]

        expected_parameters = (test_solution[corrected_parameter_fields].values)[0]
        logging.info(f"expected_parameters {expected_parameters}")
        actual_parameters = astrometric_parameter_corrected_values
        logging.info(f"actual_parameters   {actual_parameters}")
        logging.info(f"Input               {astrometric_parameter_values}")
        logging.info(f"Corrections         {astrometric_parameter_corrections}")
        assert_allclose(expected_parameters, actual_parameters)

        # compare their uncertainties
        corrected_parameter_error_fields = [f"{s}Error" for s in corrected_parameter_fields]

        # take care of naming conventions
        corrected_parameter_error_fields[corrected_parameter_error_fields.index('alphaError')] = 'alphaStarError'
        astrometric_parameter_corrections_uncertainties = results['parameters_formal_uncertainty'][0:n_linear_parameters]

        # take care of unit conventions, convert from rad to mas
        astrometric_parameter_corrections_uncertainties[0:2] *= rad2mas

        assert_allclose(test_solution[corrected_parameter_error_fields].values[0], astrometric_parameter_corrections_uncertainties)

        # compare chi2 and other metrics and number of rejected outliers
        assert_allclose(test_solution['chi2Al'], getattr(results['solution_statistic'], 'chi2'))
        assert_allclose(test_solution['f2'], getattr(results['solution_statistic'], 'f2'))

        logging.info(f"{source_id}: nObsAl={test_solution['nObsAl'].values[0]}; nOutliersAl={test_solution['nOutliersAl'].values[0]}")
        assert_allclose(test_solution['nObsAl'].values[0], getattr(results['solution_statistic'], 'n_measurements'))
        assert_allclose(test_solution['nOutliersAl'].values[0], getattr(results['solution_statistic'], 'n_outliers'))


def test_get_weighted_ssr_and_derivative():
    """Replicate the Java test testGetWeightedSsrAndDerivative.

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testGetWeightedSsrAndDerivative()

    Notes
    -----
        SSR is the "weighted sum of squared residuals"
    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalData_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    # before doing anything, check that initial normal equations are set up properly
    normals_full, rhs = design_equation.construct_full_normals()
    # reference value extracted from Java test via Eclipse debugger
    normals_full_col_5_expected = np.array([-41819.05175631906, 169755.56341404552,
                                            -259718.84479641242, 841150.0638005715,
                                            -1549765.7980301743, 4446435.407187428])
    delta = 1e-11
    # logging.info(normals_full[:, 5])
    assert_allclose(normals_full[:, 5], normals_full_col_5_expected, atol=delta)
    rhs_expected = np.array([-58575.10351659111, 135507.88624655383, -298458.8878009151,
                             716096.7543439377, -1636149.6697033513, 3944775.7503657495])
    # logging.info(rhs)
    assert_allclose(rhs, rhs_expected, atol=delta)

    parameters = np.linalg.solve(normals_full, rhs)
    residuals = design_equation.dependent_variable - design_equation.design_matrix_coefficients @ parameters
    # logging.info(residuals)
    assert_allclose(residuals[0], -0.20396639000925632, atol=delta)
    assert_allclose(df.loc[0, 'observationVariance'], 0.005310000116126305, atol=delta)

    excess_source_variance = 0
    ssr = np.sum(residuals ** 2 * df['downweight'].values * 1 / (df['observationVariance'].values + excess_source_variance))
    ssr_derivative = -1 * np.sum(residuals ** 2 * df['downweight'].values * (1 / (df['observationVariance'].values + excess_source_variance)) ** 2)

    s_expected = np.array([271.449484136445, -35905.8988702272])
    s_computed = np.array([ssr, ssr_derivative])
    logging.info(f"got      {s_computed}")
    logging.info(f"expected {s_expected}")
    assert_allclose(s_computed, s_expected, atol=delta)

    # exercise dedicated method
    s_computed2 = np.array(design_equation.get_weighted_ssr_and_derivative())
    assert_allclose(s_computed, s_computed2, atol=delta)


def test_get_calculate_excess_source_noise():
    """Replicate the Java test RobustSourceUpdateCalculatorTest.testCalculateExcessSourceNoise().

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testCalculateExcessSourceNoise()
    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalData_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    en_expected = [0.223400837560798, 25.8768389943175]

    delta = 1e-11
    en_computed = np.array(design_equation.calculate_excess_source_noise())
    logging.info(f"got      {en_computed}")
    logging.info(f"expected {en_expected}")
    assert_allclose(en_computed, en_expected, atol=delta)


def test_calculate_downweights():
    """Replicate the Java test RobustSourceUpdateCalculatorTest.testCalculateDownweights().

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testCalculateDownweights()
    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalDataWithOutlier_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    # before doing anything, check that initial normal equations are set up properly
    normals_full, rhs = design_equation.construct_full_normals()

    delta = 4e-7

    downweights_expected = np.array([0.0867119356841193,
                                     0.350446649755513,
                                     0.00418868642330765,
                                     0.257524891564308,
                                     0.484825645005833,
                                     0.186782877418076,
                                     0.335697474803835,
                                     1.0,
                                     0.269725705330574,
                                     1.0,
                                     1.0,
                                     1.0,
                                     0.670743990999004,
                                     1.0,
                                     0.778806592191732,
                                     0.329525522887946,
                                     0.304510147716451,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     0.902764122785385,
                                     0.363126204142606,
                                     0.277076500769437,
                                     0.30891353292887,
                                     0.422362445310823,
                                     0.885099709979087,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     1.0,
                                     0.973223274001493,
                                     0.183944109883559,
                                     0.223257076299538,
                                     0.944576413628919,
                                     1.0,
                                     1.0,
                                     1.0])

    logging.debug(f"Expecting {len(downweights_expected)} downweights")
    parameters = np.linalg.solve(normals_full, rhs)
    residuals = design_equation.dependent_variable - design_equation.design_matrix_coefficients @ parameters
    logging.debug(f"residuals = {residuals}")

    normalised_residuals = residuals / np.sqrt(df['observationVariance'] + df['excessSourceVariance'])
    logging.debug(f"normalised_residuals = {normalised_residuals}")

    # downweightPrior is always 1 in this dataset
    downweights = decay_downweight(normalised_residuals)
    assert_allclose(downweights, downweights_expected, atol=delta)


def test_get_n_outlier():
    """Replicate the Java test RobustSourceUpdateCalculatorTest.testGetNOut().

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testGetNOut()
    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalDataWithOutlier_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    number_of_outliers_expected = 4

    residuals = design_equation.get_postfit_residuals()
    downweights = design_equation.calculate_downweights(residuals)
    design_equation.weight = downweights

    number_of_outliers_computed = design_equation.compute_n_outliers()

    logging.debug(f"number_of_outliers_expected = {number_of_outliers_expected}")
    logging.debug(f"number_of_outliers_computed = {number_of_outliers_computed}")
    assert_allclose(number_of_outliers_expected, number_of_outliers_computed)


def test_calculate_robust_estimate_1():
    """Replicate the Java test RobustSourceUpdateCalculatorTest.testCalculateRobustEstimate1().

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testCalculateRobustEstimate1()
    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalDataWithOutlier_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    excess_source_noise, excess_source_noise_significance = design_equation.calculate_estimate()

    en_expected = [0.240355624324242, 29.7198136375806]
    en_computed = np.array([excess_source_noise, excess_source_noise_significance])

    # delta = 4e-7  # Java against Matlab threshold
    delta = 1e-11  # effectively python versus Matlab threshold
    logging.info(f"got      {en_computed}")
    logging.info(f"expected {en_expected}")
    assert_allclose(en_computed, en_expected, atol=delta)

    updates_expected = np.array([-0.989880471516826,
                                 0.389512527826394,
                                 1.20360663330997,
                                 -0.230316839281475,
                                 -1.10234187473455,
                                 0.591645645517887])
    updates_computed = design_equation.get_updates()
    logging.info(f"got      {updates_computed}")
    logging.info(f"expected {updates_expected}")
    assert_allclose(updates_computed, updates_expected, atol=delta)

    # exercise solve_like_agis
    design_equation2 = DesignEquation(design_parameters)
    results = design_equation2.solve(solver='agis')
    assert_allclose(results['parameters'], updates_expected, atol=delta)
    assert_allclose(results['excess_noise'], en_expected[0], atol=delta)
    assert_allclose(results['significance'], en_expected[1], atol=delta)


def test_calculate_robust_estimate_2():
    """Replicate the Java test RobustSourceUpdateCalculatorTest.testCalculateRobustEstimate2().

    See gaia.cu3.agistools.algo.gis.source.test.RobustSourceUpdateCalculatorTest.testCalculateRobustEstimate2()

    """
    coefficients_file = 'RobustSourceUpdateCalculatorTest_setupNominalDataWithOutliers_coefficients.csv'
    logging.info(f'Using input file {coefficients_file}')
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', coefficients_file)
    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info(f'number of rows {len(df)}')
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    excess_source_noise, excess_source_noise_significance = design_equation.calculate_estimate()

    en_expected = [0.311563820854682, 46.1515024313121]
    en_computed = np.array([excess_source_noise, excess_source_noise_significance])

    delta = 6e-4
    logging.info(f"got      {en_computed}")
    logging.info(f"expected {en_expected}")
    assert_allclose(en_computed, en_expected, atol=delta)

    delta = 7e-7
    updates_expected = np.array([-1.00323655503195,
                                 0.393040111876353,
                                 1.24712375171721,
                                 -0.216294799828578,
                                 -1.11283053160205,
                                 0.587411559911783])
    updates_computed = design_equation.get_updates()
    logging.info(f"got      {updates_computed}")
    logging.info(f"expected {updates_expected}")
    assert_allclose(updates_computed, updates_expected, atol=delta)

    logging.info(design_equation.n_solve)


def test_get_updates():
    """Replicate the Java test BasicSourceUpdateCalculatorTest.testGetUpdates().

    See gaia.cu3.agistools.algo.gis.source.test.BasicSourceUpdateCalculatorTest.testGetUpdates().

    This tests against dataset used in
    /AGISTools/src/gaia/cu3/agistools/algo/gis/source/test/BasicSourceUpdateCalculatorTest.java

    See also test_against_basic_source_update

    """
    data_file = os.path.join(TEST_DATA_ROOT, 'agis', 'BasicSourceUpdateCalculatorTest_setupNominalData_coefficients.csv')

    # df = DpacDataFrame.from_csv_file(data_file)
    df = pd.read_csv(data_file, skipinitialspace=True)
    logging.info('')
    logging.info(df.columns)
    selected_columns = [c for c in df.columns if 'coefficients_' in c]

    design_parameters = OrderedDict()
    design_parameters['design_matrix_coefficients'] = df[selected_columns].to_numpy()
    design_parameters['normal_matrix_column_names'] = selected_columns
    design_parameters['dependent_variable'] = df['rhs_value'].values
    design_parameters['dependent_variable_error'] = df['rhs_sigma'].values
    design_parameters['weight'] = df['downweight'].values
    design_parameters['model'] = 'test'
    design_equation = DesignEquation(design_parameters)

    x_computed = design_equation.get_updates()

    # according to testGetUpdates() // nominal data (MATLAB case 0)
    x_expected = np.array([-0.994944027626899,
                           0.511899022564177,
                           1.18509074022589,
                           -0.300448784065878,
                           -1.09789066546504,
                           0.599632828381641])
    delta = 1e-11
    logging.info(f"got      {x_computed}")
    logging.info(f"expected {x_expected}")
    assert_allclose(x_computed, x_expected, atol=delta)


def test_decay_downweight():
    """Test decay downweight calculation."""
    z = [-5.00000000000000, -4.90000000000000, -4.80000000000000,
         -4.70000000000000, -4.60000000000000, -4.50000000000000,
         -4.40000000000000, -4.30000000000000, -4.20000000000000,
         -4.10000000000000, -4.00000000000000, -3.90000000000000,
         -3.80000000000000, -3.70000000000000, -3.60000000000000,
         -3.50000000000000, -3.40000000000000, -3.30000000000000,
         -3.20000000000000, -3.10000000000000, -3.00000000000000,
         -2.90000000000000, -2.80000000000000, -2.70000000000000,
         -2.60000000000000, -2.50000000000000, -2.40000000000000,
         -2.30000000000000, -2.20000000000000, -2.10000000000000,
         -2.00000000000000, -1.90000000000000, -1.80000000000000,
         -1.70000000000000, -1.60000000000000, -1.50000000000000,
         -1.40000000000000, -1.30000000000000, -1.20000000000000,
         -1.10000000000000, -1.00000000000000, -0.90000000000000,
         -0.80000000000000, -0.70000000000000, -0.60000000000000,
         -0.50000000000000, -0.40000000000000, -0.30000000000000,
         -0.20000000000000, -0.10000000000000, 0.00000000000000,
         0.10000000000000, 0.20000000000000, 0.30000000000000,
         0.40000000000000, 0.50000000000000, 0.60000000000000,
         0.70000000000000, 0.80000000000000, 0.90000000000000,
         1.00000000000000, 1.10000000000000, 1.20000000000000,
         1.30000000000000, 1.40000000000000, 1.50000000000000,
         1.60000000000000, 1.70000000000000, 1.80000000000000,
         1.90000000000000, 2.00000000000000, 2.10000000000000,
         2.20000000000000, 2.30000000000000, 2.40000000000000,
         2.50000000000000, 2.60000000000000, 2.70000000000000,
         2.80000000000000, 2.90000000000000, 3.00000000000000,
         3.10000000000000, 3.20000000000000, 3.30000000000000,
         3.40000000000000, 3.50000000000000, 3.60000000000000,
         3.70000000000000, 3.80000000000000, 3.90000000000000,
         4.00000000000000, 4.10000000000000, 4.20000000000000,
         4.30000000000000, 4.40000000000000, 4.50000000000000,
         4.60000000000000, 4.70000000000000, 4.80000000000000,
         4.90000000000000]

    expected_weight = [0.18887560283756, 0.19527756283569, 0.20189651799466,
                       0.20873982339008, 0.21581508339869, 0.22313016014843,
                       0.23069318225496, 0.23851255385430, 0.24659696394161,
                       0.25495539602651, 0.26359713811573, 0.27253179303401,
                       0.28176928909496, 0.29131989113347, 0.30119421191220,
                       0.31140322391460, 0.32195827153768, 0.33287108369808,
                       0.34415378686541, 0.35581891853734, 0.36787944117144,
                       0.39551156173027, 0.44931616877959, 0.52244357449581,
                       0.60804409105532, 0.69926803063453, 0.78926570540983,
                       0.87118742755763, 0.93818350925433, 0.98340426267631,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 0.98340426267631,
                       0.93818350925433, 0.87118742755763, 0.78926570540983,
                       0.69926803063453, 0.60804409105532, 0.52244357449581,
                       0.44931616877959, 0.39551156173027, 0.36787944117144,
                       0.35581891853734, 0.34415378686541, 0.33287108369808,
                       0.32195827153768, 0.31140322391460, 0.30119421191220,
                       0.29131989113347, 0.28176928909496, 0.27253179303401,
                       0.26359713811573, 0.25495539602651, 0.24659696394161,
                       0.23851255385430, 0.23069318225496, 0.22313016014843,
                       0.21581508339869, 0.20873982339008, 0.20189651799466,
                       0.19527756283569]

    scale_factor = 1.
    computed_weight = decay_downweight(z, scale_factor=scale_factor)
    assert_allclose(expected_weight, computed_weight)

    scale_factor = 0.5
    expected_weight = [0.69926803063453, 0.74484825661165, 0.78926570540983,
                       0.83166416605115, 0.87118742755763, 0.90697927895134,
                       0.93818350925433, 0.96394390748863, 0.98340426267631,
                       0.99570836383942, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 0.99570836383942, 0.98340426267631,
                       0.96394390748863, 0.93818350925433, 0.90697927895134,
                       0.87118742755763, 0.83166416605115, 0.78926570540983,
                       0.74484825661165]

    computed_weight = decay_downweight(z, scale_factor=scale_factor)
    assert_allclose(expected_weight, computed_weight)


def test_huber_downweight():
    """Test the Huber downweight calculation."""
    # z = np.linspace(-5, 5, 100)
    z = [-5.00000000000000, -4.90000000000000, -4.80000000000000,
         -4.70000000000000, -4.60000000000000, -4.50000000000000,
         -4.40000000000000, -4.30000000000000, -4.20000000000000,
         -4.10000000000000, -4.00000000000000, -3.90000000000000,
         -3.80000000000000, -3.70000000000000, -3.60000000000000,
         -3.50000000000000, -3.40000000000000, -3.30000000000000,
         -3.20000000000000, -3.10000000000000, -3.00000000000000,
         -2.90000000000000, -2.80000000000000, -2.70000000000000,
         -2.60000000000000, -2.50000000000000, -2.40000000000000,
         -2.30000000000000, -2.20000000000000, -2.10000000000000,
         -2.00000000000000, -1.90000000000000, -1.80000000000000,
         -1.70000000000000, -1.60000000000000, -1.50000000000000,
         -1.40000000000000, -1.30000000000000, -1.20000000000000,
         -1.10000000000000, -1.00000000000000, -0.90000000000000,
         -0.80000000000000, -0.70000000000000, -0.60000000000000,
         -0.50000000000000, -0.40000000000000, -0.30000000000000,
         -0.20000000000000, -0.10000000000000, 0.00000000000000,
         0.10000000000000, 0.20000000000000, 0.30000000000000,
         0.40000000000000, 0.50000000000000, 0.60000000000000,
         0.70000000000000, 0.80000000000000, 0.90000000000000,
         1.00000000000000, 1.10000000000000, 1.20000000000000,
         1.30000000000000, 1.40000000000000, 1.50000000000000,
         1.60000000000000, 1.70000000000000, 1.80000000000000,
         1.90000000000000, 2.00000000000000, 2.10000000000000,
         2.20000000000000, 2.30000000000000, 2.40000000000000,
         2.50000000000000, 2.60000000000000, 2.70000000000000,
         2.80000000000000, 2.90000000000000, 3.00000000000000,
         3.10000000000000, 3.20000000000000, 3.30000000000000,
         3.40000000000000, 3.50000000000000, 3.60000000000000,
         3.70000000000000, 3.80000000000000, 3.90000000000000,
         4.00000000000000, 4.10000000000000, 4.20000000000000,
         4.30000000000000, 4.40000000000000, 4.50000000000000,
         4.60000000000000, 4.70000000000000, 4.80000000000000,
         4.90000000000000]

    expected_weight = [0.64000000000000, 0.64972927946689, 0.65972222222222, 0.66998641919421,
                       0.68052930056711, 0.69135802469136, 0.70247933884298, 0.71389940508383,
                       0.72562358276644, 0.73765615704938, 0.75000000000000, 0.76265614727153,
                       0.77562326869806, 0.78889700511322, 0.80246913580247, 0.81632653061224,
                       0.83044982698962, 0.84481175390266, 0.85937500000000, 0.87408949011446,
                       0.88888888888889, 0.90368608799049, 0.91836734693878, 0.93278463648834,
                       0.94674556213018, 0.96000000000000, 0.97222222222222, 0.98298676748582,
                       0.99173553719008, 0.99773242630386, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000, 0.99773242630386,
                       0.99173553719008, 0.98298676748582, 0.97222222222222, 0.96000000000000,
                       0.94674556213018, 0.93278463648834, 0.91836734693878, 0.90368608799049,
                       0.88888888888889, 0.87408949011446, 0.85937500000000, 0.84481175390266,
                       0.83044982698962, 0.81632653061224, 0.80246913580247, 0.78889700511322,
                       0.77562326869806, 0.76265614727153, 0.75000000000000, 0.73765615704938,
                       0.72562358276644, 0.71389940508383, 0.70247933884298, 0.69135802469136,
                       0.68052930056711, 0.66998641919421, 0.65972222222222, 0.64972927946689]

    scale_factor = 1.
    computed_weight = huber_downweight(z, scale_factor=scale_factor)
    assert_allclose(expected_weight, computed_weight)

    scale_factor = 0.5
    expected_weight = [0.96000000000000, 0.96626405664307, 0.97222222222222,
                       0.97781801720235, 0.98298676748582, 0.98765432098765,
                       0.99173553719008, 0.99513250405625, 0.99773242630386,
                       0.99940511600238, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 1.00000000000000, 1.00000000000000,
                       1.00000000000000, 0.99940511600238, 0.99773242630386,
                       0.99513250405625, 0.99173553719008, 0.98765432098765,
                       0.98298676748582, 0.97781801720235, 0.97222222222222,
                       0.96626405664307]

    computed_weight = huber_downweight(z, scale_factor=scale_factor)
    assert_allclose(expected_weight, computed_weight)
