#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the source update using GaiaSourceEpochAstrometry."""

__author__ = "Johannes Sahlmann"

import logging
import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryCu9, GaiaEpochAstrometryCu9

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_epoch_astrometry_source_update():
    """Test the source update on the basis of epoch astrometry in parquet format."""
    # read epoch astrometry data
    file = os.path.join(TEST_DATA_ROOT, 'epoch_astrometry_sample.parquet')
    epoch_astrometry_df = pd.read_parquet(file)

    # Load the data into the appropriate GaiaEpochAstrometry object, which flattens the array columns etc.
    epoch_astrometry_all = GaiaEpochAstrometryCu9.from_dataframe(epoch_astrometry_df)

    # select one source and its data
    selected_source_id = epoch_astrometry_all.epoch_data.loc[0, 'sourceId']
    epoch_astrometry_df = epoch_astrometry_all.epoch_data[epoch_astrometry_all.epoch_data['sourceId'] == selected_source_id]

    # load data into appropriate GaiaSourceEpochAstrometry object
    sea = GaiaSourceEpochAstrometryCu9.from_dataframe(epoch_astrometry_df, selected_source_id, is_exploded=True)

    # select only measurements used by AGIS
    sea.epoch_data = sea.epoch_data.epochastrometrycu9.filter_on_used_by_agis()

    # compute the source update and check results
    source_update_results = sea.compute_source_update()
    assert source_update_results['success'] == 1

    expected_parameters = np.array([-3.65476422e-03, -3.68525745e-04, 1.28540208e+00, 3.65254875e-01, -4.66684150e+00])
    actual_parameters = source_update_results['results']['parameters']
    assert_allclose(actual_parameters, expected_parameters)


def test_double_initialisation():
    """Test that the input dataframe is not modified."""
    # read epoch astrometry data
    file = os.path.join(TEST_DATA_ROOT, 'epoch_astrometry_sample.parquet')
    epoch_astrometry_df = pd.read_parquet(file)

    epoch_astrometry_df_original = epoch_astrometry_df
    # Load the data into the appropriate GaiaEpochAstrometry object, which flattens the array columns etc.
    pd.testing.assert_frame_equal(epoch_astrometry_df, epoch_astrometry_df_original)
