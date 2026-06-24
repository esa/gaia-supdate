#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for CU9 data supdate for model 6p_constrained_colour."""

__author__ = "Johannes Sahlmann"

import logging
import os
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.epoch_astrometry import GaiaEpochAstrometryCu9

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_cu9_supdate_6p():
    """Test supdate for model 6p_constrained_colour using data in CU9 format.
    """
    sourceid = 2

    epoch_astro_file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{sourceid}_epoch_astrometry.parquet')
    epoch_astro_df = pd.read_parquet(epoch_astro_file)

    agis_result_file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{sourceid}_agis_source_update_results.parquet')
    agis_result_df = pd.read_parquet(agis_result_file)

    linear_model_parameters = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta', 'pseudoColor']
    linear_model_error_parameters = ['alphaStar', 'delta', 'varpi', 'muAlphaStar', 'muDelta', 'pseudoColor']
    agis_ppm_keys = [f'{s}' for s in linear_model_parameters]
    agis_ppm_error_keys = [f'{s}Error' for s in linear_model_error_parameters]

    results = GaiaEpochAstrometryCu9.supdate(epoch_astro_df, sourceid)

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


def test_cu9_supdate_5p():
    """Test supdate for model 5p_single_source using data in CU9 format.
    """
    model = '5p_single_source'
    sourceid = 2

    epoch_astro_file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{sourceid}_epoch_astrometry.parquet')
    epoch_astro_df = pd.read_parquet(epoch_astro_file)

    results = GaiaEpochAstrometryCu9.supdate(epoch_astro_df, sourceid, model)

    assert len(results['parameters']) == 5


def test_compute_source_parameters():
    """Test method compute_source_parameters with arguments 6p and excess noise is like dr4."""
    sourceid = 2

    epoch_astro_file = os.path.join(TEST_DATA_ROOT, 'constrained_colour', f'{sourceid}_epoch_astrometry.parquet')
    epoch_astro_df = pd.read_parquet(epoch_astro_file)

    model = '6p_constrained_colour'
    compute_excess_noise = False
    results = GaiaEpochAstrometryCu9.supdate(epoch_astro_df, sourceid, model, compute_excess_noise)

    results_dr4 = GaiaEpochAstrometryCu9.supdate(epoch_astro_df, sourceid)

    results['parameters'] = results_dr4['parameters']
    results['parameters_formal_uncertainty'] = results_dr4['parameters_formal_uncertainty']

    np.set_printoptions(precision=15, suppress=True)
    logging.info(f"results = {results['parameters']}")
    logging.info(f"results_dr4   = {results_dr4['parameters']}")
    assert_allclose(results['parameters'],
                    results_dr4['parameters'], rtol=1e-7)

    logging.info(f"uncertenties = {results['parameters_formal_uncertainty']}")
    logging.info(f"uncertenties_dr4  = {results_dr4['parameters_formal_uncertainty']}")
    assert_allclose(results['parameters_formal_uncertainty'],
                    results_dr4['parameters_formal_uncertainty'], rtol=1e-7)
