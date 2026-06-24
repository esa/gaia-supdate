#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the source update using GaiaSourceEpochAstrometry using the GACS interface file formats."""

__author__ = "Johannes Sahlmann"

import logging
import os
import sys

from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from tests.constants import TEST_DATA_ROOT
from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryArchive, GaiaEpochAstrometryArchive
from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryCu9, GaiaEpochAstrometryCu9

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@pytest.mark.parametrize('dataformat', ['csv', 'ecsv', 'xml', 'votplain.xml', 'fits', 'parquet'])
@pytest.mark.skipif((sys.version_info[0], sys.version_info[1]) == (3, 7), reason="ecsv cannot be read.")
def test_gacs_interface_single_source(dataformat):
    """Test the source update on the basis of epoch astrometry in parquet format."""

    if dataformat in ['parquet']:
        file = os.path.join(TEST_DATA_ROOT, 'gacs_data_link', f'1_epoch_astrometry.{dataformat}')
        epoch_astrometry_df = pd.read_parquet(file)
    else:
        file = os.path.join(TEST_DATA_ROOT, 'gacs_data_link', f'EPOCH_ASTROMETRY-Gaia DR4_INT2 1.{dataformat}')
        df = Table.read(file).to_pandas()
        epoch_astrometry_df = GaiaEpochAstrometryArchive.astropy_table_to_df(df)

    if dataformat in ['fits']:
        # handle boolean columns
        epoch_astrometry_df['used_by_agis_al'] = epoch_astrometry_df['used_by_agis_al'].apply(lambda x: [a == 84 for a in x])
        epoch_astrometry_df['used_by_agis_ac'] = epoch_astrometry_df['used_by_agis_ac'].apply(lambda x: [a == 84 for a in x])

    # Load the data into the appropriate GaiaEpochAstrometry object, which flattens the array columns etc.
    if dataformat in ['parquet']:
        epoch_astrometry_all = GaiaEpochAstrometryCu9.from_dataframe(epoch_astrometry_df)
    else:
        epoch_astrometry_all = GaiaEpochAstrometryArchive.from_dataframe(epoch_astrometry_df)

    # select one source and its data  epoch_astrometry_all._source_id_field
    selected_source_id = epoch_astrometry_all.epoch_data.loc[0, epoch_astrometry_all._source_id_field]
    epoch_astrometry_df = epoch_astrometry_all.epoch_data[epoch_astrometry_all.epoch_data[epoch_astrometry_all._source_id_field] == selected_source_id]

    # load data into appropriate GaiaSourceEpochAstrometry object and select only measurements used by AGIS
    if dataformat in ['parquet']:
        sea = GaiaSourceEpochAstrometryCu9.from_dataframe(epoch_astrometry_df, selected_source_id, is_exploded=True)
        sea.epoch_data = sea.epoch_data.epochastrometrycu9.filter_on_used_by_agis()
    else:
        sea = GaiaSourceEpochAstrometryArchive.from_dataframe(epoch_astrometry_df, selected_source_id, is_exploded=True)
        sea.epoch_data = sea.epoch_data.epochastrometryarchive.filter_on_used_by_agis()

    # compute the source update and check results
    source_update_results = sea.compute_source_update()
    assert source_update_results['success'] == 1

    if dataformat in ['csv']:
        expected_parameters = np.array([-1.48336910e-03, -3.85162037e-03, 3.07244859e+00, -9.89371411e+00, 6.01353995e+00])
    elif dataformat in ['votplain.xml']:
        expected_parameters = np.array([-1.48337421e-03, -3.85162116e-03, 3.07244857e+00, -9.89371411e+00, 6.01353995e+00])
    else:
        expected_parameters = np.array([-1.48337410e-03, -3.85162113e-03, 3.07244857e+00, -9.89371411e+00, 6.01353995e+00])

    actual_parameters = source_update_results['results']['parameters']
    logging.info(actual_parameters)
    assert_allclose(actual_parameters, expected_parameters, rtol=4e-8)
