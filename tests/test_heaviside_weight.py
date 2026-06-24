#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the epoch astrometry source update with Heaviside weights."""

__author__ = "Johannes Sahlmann"

import logging
import os

from numpy.testing import assert_allclose
import pandas as pd

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryCu9, GaiaEpochAstrometryCu9
from gaiasupdate.solver import DesignEquation

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_heaviside_source_update():
    """Test the source update with Heaviside weighting.

    The reference data were obtained with AGIS and setting gaia.cu3.agis.useHeavisideWeights=true.

    """
    # read epoch astrometry data
    file = os.path.join(TEST_DATA_ROOT, 'heaviside_weighting', 'epoch_astrometry.parquet')
    epoch_astrometry_df = pd.read_parquet(file)

    # Load the data into the appropriate GaiaEpochAstrometry object, which flattens the array columns etc.
    epoch_astrometry_all = GaiaEpochAstrometryCu9.from_dataframe(epoch_astrometry_df)

    # load the reference dataset. That is needed to extract the AGIS excess nource noise and for comparison.
    agis_result_df = pd.read_parquet(os.path.join(TEST_DATA_ROOT, 'heaviside_weighting', 'agis_source_update_results.parquet'))
    agis_excess_noise_dict = agis_result_df.set_index('sourceId')['excessNoise_mas'].to_dict()

    epoch_astrometry_grouped = epoch_astrometry_all.epoch_data.groupby('sourceId')

    #  model definition
    use_heaviside_weights = True
    model = '5p_single_source'

    # helper dictionaries
    map_source_to_results = {'excessNoise': 'excess_noise', 'excessNoiseSig': 'significance', }
    map_source_to_statistic = {'f2': 'f2', 'chi2Al': 'chi2', 'nObsAl': 'n_measurements',
                               'nOutliersAl': 'n_outliers', }
    if model == '5p_single_source':
        linear_model_parameters = ['alpha', 'delta', 'varpi', 'muAlphaStar', 'muDelta']
        linear_model_error_parameters = ['alphaStar', 'delta', 'varpi', 'muAlphaStar', 'muDelta']
    agis_ppm_keys = [f'{s}' for s in linear_model_parameters]
    agis_ppm_error_keys = [f'{s}Error' for s in linear_model_error_parameters]

    #  loop over sources
    result_parameters_all = []
    for selected_source_id, source_df in epoch_astrometry_grouped:
        result_parameters = {'sourceId': selected_source_id}

        sea = GaiaSourceEpochAstrometryCu9.from_dataframe(source_df, selected_source_id, is_exploded=True)

        # use only data that were also used by AGIS
        sea.epoch_data = sea.epoch_data.epochastrometrycu9.filter_on_used_by_agis()

        #  perform standard adjustments to dataframe
        sea.epoch_data.epochastrometrycu9.set_relative_time()

        # get design equation
        design_parameters = sea.epoch_data.epochastrometrycu9.get_design_equation_parameters(model=model)
        design_equation = DesignEquation(design_parameters)
        solver = 'agis'

        # compute solution
        if use_heaviside_weights:
            # when using Heaviside weighting, the python code needs to be given the total variance.
            # Total variance is then kept fixed and a simple linear least-square solution is computed.
            # The source excess noise is set to the AGIS value, in agreement with the AGIS solution.
            excess_source_noise_mas = agis_excess_noise_dict[selected_source_id]
            excess_source_variance = excess_source_noise_mas**2
            total_variance = design_equation.observation_variances + excess_source_variance
        else:
            total_variance = None

        results = design_equation.solve(solver=solver, total_variance=total_variance)
        result_parameters['success'] = 1

        # extract solution fields with AGIS-like naming
        for field, mapped_field in map_source_to_results.items():
            result_parameters[field] = results[mapped_field]
        for field, mapped_field in map_source_to_statistic.items():
            result_parameters[field] = getattr(results['solution_statistic'], mapped_field)
        for i, key in enumerate(agis_ppm_error_keys):
            factor = 1
            result_parameters[key] = results['parameters_formal_uncertainty'][i] * factor
        for i, key in enumerate(agis_ppm_keys):
            result_parameters[key] = results['parameters'][i]

        result_parameters_all.append(result_parameters)

    # convert results to dataframe
    epoch_astrometry_result_df = pd.DataFrame(result_parameters_all)
    logging.info(f"Input sourcegroups have {epoch_astrometry_df['sourceId'].nunique()} sources")
    epoch_astrometry_result_df = epoch_astrometry_result_df[epoch_astrometry_result_df['success'] == 1]
    logging.info(f"epoch_astrometry_result_df has {len(epoch_astrometry_result_df)} rows (success==1)")
    logging.info(f"epoch_astrometry_result_df has {epoch_astrometry_result_df['sourceId'].nunique()} unique sourceIds")

    # match python solutions with AGIS solution parameters
    tdf = agis_result_df.merge(epoch_astrometry_result_df, on='sourceId', suffixes=('_agis', '_ea'))
    logging.info(f"tdf has {len(tdf)} rows ")

    # make sure the number of used observations is the same in both cases
    tdf = tdf[tdf['nObsAl_agis'] == tdf['nObsAl_ea']]
    logging.info(f"Filter on nObsAl leaves {len(tdf)} rows")

    # check parallax difference
    for key in ['varpi', 'varpiError', 'muAlphaStar']:
        key0 = f"{key}_agis"
        key1 = f"{key}_ea"
        tdf['discrepancy'] = tdf[key1] - tdf[key0]
        logging.info(f"\n{tdf[['sourceId', 'gMag', 'nuEff', 'nObsAl_agis', 'discrepancy']]}")
        if key == 'varpi':
            assert_allclose(tdf['discrepancy'].values, 0, atol=1e-2)
            # agreement is worse for bright sources
            assert_allclose(tdf[tdf['gMag'] > 7]['discrepancy'].values, 0, atol=2e-4)
        elif key == 'varpiError':
            assert_allclose(tdf['discrepancy'].values, 0, atol=2e-6)
        elif key == 'muAlphaStar':
            assert_allclose(tdf[tdf['gMag'] > 7]['discrepancy'].values, 0, atol=1e-4)
