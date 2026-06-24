#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the constrained-colour source update."""

__author__ = "Johannes Sahlmann"

import logging
import os
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryCu9, GaiaEpochAstrometryCu9
from gaiasupdate.solver import DesignEquation

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_constrained_colour_source_update():
    """Test the source update with Heaviside weighting.

    The reference data were obtained with AGIS and setting gaia.cu3.agis.useHeavisideWeights=true.

    """
    sourceid = 2
    # read epoch astrometry data
    file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{sourceid}_epoch_astrometry.parquet')
    epoch_astrometry_df = pd.read_parquet(file)

    agis_result_df = pd.read_parquet(os.path.join(TEST_DATA_ROOT, 'constrained_colour',
                                                  f'{sourceid}_agis_source_update_results.parquet'))

    linear_model_parameters = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta', 'pseudoColor']
    linear_model_error_parameters = ['alphaStar', 'delta', 'varpi', 'muAlphaStar', 'muDelta', 'pseudoColor']
    agis_ppm_keys = [f'{s}' for s in linear_model_parameters]
    agis_ppm_error_keys = [f'{s}Error' for s in linear_model_error_parameters]

    epoch_astrometry_all = GaiaEpochAstrometryCu9.from_dataframe(epoch_astrometry_df)

    # the native unit of H01Correction is TDI*nm, but in LPCcentroid and EpochAstrometry it has been converted to mas*mum
    # for this test dataset, the unit of nuEff, pseudoColour etc. is in 1/nm

    # therefore we convert the factors here from mas*mum -> mas*nm
    # the -1 factor originates in the -1 factor used in
    # gaia.cu3.agis.algo.gis.sourceimpl.SourceUpdateCalculatorWrapper.calcUpdate:
    # dEtaZeta[length] = -xCalibration[stripIdx];
    epoch_astrometry_all.epoch_data['colourFactorAl'] *= -1e3
    epoch_astrometry_grouped = epoch_astrometry_all.epoch_data.groupby('sourceId')

    model = '6p_constrained_colour'

    filter_on_used_by_agis = True
    for selected_source_id, source_df in epoch_astrometry_grouped:
        sea = GaiaSourceEpochAstrometryCu9.from_dataframe(source_df, selected_source_id, is_exploded=True)

        if 0:
            if filter_on_used_by_agis:
                sea.epoch_data = sea.epoch_data.epochastrometrycu9.filter_on_used_by_agis()

            sea.epoch_data = sea.epoch_data.gaiacentroid.filter_null_from_column(sea.epoch_data.epochastrometrycu9._time_barycentric_correction_column)
            sea.epoch_data = sea.epoch_data.gaiacentroid.filter_null_from_column(sea.epoch_data.epochastrometrycu9._scan_angle_column)
            sea.epoch_data.epochastrometrycu9.set_relative_time()

            logging.info(sea)
            design_parameters = sea.epoch_data.epochastrometrycu9.get_design_equation_parameters(model=model)

            n_linear_parameters = design_parameters['design_matrix_coefficients'].shape[1]
            if model == '5p_single_source':
                assert n_linear_parameters == 5
            else:
                assert n_linear_parameters == 6
                prior_strength_nm = 0.085e-3  # this is in units of 1/nm
                design_parameters['gaussian_priors'] = np.array([None] * (n_linear_parameters - 1) + [prior_strength_nm])

            design_parameters['model'] = model

            design_equation = DesignEquation(design_parameters)
            excess_source_noise_mas = sea.epoch_data.iloc[0][sea._agis_source_excess_noise_column]

            excess_source_variance = excess_source_noise_mas**2
            total_variance = design_equation.observation_variances + excess_source_variance

            solver = 'agis'
            results = design_equation.solve(solver=solver, total_variance=total_variance)
        else:
            results = sea.compute_source_parameters_like_dr4()

        logging.info(results['solution_statistic'])
        logging.info(f"AGIS chi2Al   = {agis_result_df.loc[0, 'chi2Al']}")

        expected_results = agis_result_df.loc[0, agis_ppm_keys].values.astype(float)
        expected_uncertainties = agis_result_df.loc[0, agis_ppm_error_keys].values.astype(float)
        actual_results = results['parameters']
        actual_uncertainties = results['parameters_formal_uncertainty']

        np.set_printoptions(precision=15, suppress=True)
        logging.info(f"expected_results = {expected_results[2:]}")
        logging.info(f"actual_results   = {actual_results[2:]}")
        assert_allclose(expected_results[2:], actual_results[2:], rtol=3e-4)

        logging.info(f"expected_uncertainties = {expected_uncertainties}")
        logging.info(f"actual_uncertainties   = {actual_uncertainties}")
        assert_allclose(expected_uncertainties, actual_uncertainties, rtol=1e-4)
        assert agis_result_df.loc[0, 'nObsAl'] == results['n_measurements']
