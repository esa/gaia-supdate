#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Module to facilitate the handling of Gaia Epoch Astrometry."""

__author__ = "Johannes Sahlmann"

import copy
from collections import OrderedDict
from numpy.linalg import LinAlgError
import logging
import astropy.units as u
from astroquery.gaia import GaiaClass, Gaia
import pandas as pd
import numpy as np
from pandas.api.types import is_list_like

from .constants import TCB_REFERENCE_EPOCH, DR4_REFERENCE_EPOCH
from .solver import DesignEquation

from numpy import nan as NaN
false = False
true = True
mas_to_rad = u.milliarcsecond.to(u.rad)


@pd.api.extensions.register_dataframe_accessor("gaiacentroid")
class GaiaCentroidDataFrame:
    """Extension for Gaia epoch astrometry data from centroids."""

    _validation_columns = []
    _blended_column = None

    def __init__(self, pandas_obj):
        """
        Set _obj attribute and validate input dataframe.

        Parameters
        ----------
        pandas_obj: pandas dataframe
            Dataframe containing epoch astrometry data and centroids.

        """
        self._obj = pandas_obj
        for key in ['units']:
            if key not in self._obj.attrs:
                self._obj.attrs[key] = None
        self._validate(pandas_obj, self._validation_columns)

    @staticmethod
    def _validate(obj, columns):
        """Verify the necessary columns are present.

        Parameters
        ----------
        obj: pandas dataframe
            Dataframe.
        columns: list
            List of columns to be present in 'obj'.

        """
        missing_columns = np.setdiff1d(columns, obj.columns)
        if len(missing_columns) > 0:
            raise AttributeError('DataFrame must have {} columns.\nIt has {}'.format(missing_columns, obj.columns))

    def fix_blended_column(self):
        """Fix the 'blended' column, filling it with NaN and set as float type."""
        self._obj['blended'] = self._obj[self._blended_column].fillna(np.nan)
        self._obj = self._obj.astype({self._blended_column: float})

    @staticmethod
    def explode_ccdlevel_columns(df_input):
        """Convert the columns holding arrays of CCD-transit information into dataframe rows, also fixing issues \
        in certain columns.

        Note
        ----
        A new column 'component_index' is introduced to keep track of the device where the CCD transit was recorded.

        Parameters
        ----------
        df_input: pandas dataframe
            Dataframe containing GaiaCentroidDataFrame.

        Returns
        -------
        Pandas dataframe
            Dataframe with rows exploded to hold one device data per row.

        """
        df = df_input
        # to remove column 'centroid_pos_ac' or any column with all rows set to NaN
        df.dropna(axis='columns', how='all', inplace=True)
        # identify columns that hold arrays
        array_colums = [c for c, d in df.dtypes.items() if d == 'object']

        problematic_columns = ['filename', 'blended', 'multipeak']
        for col in problematic_columns:
            if col in array_colums:
                logging.warning(f"Dataframe contains column '{col}', which can cause problems \
                when exploding the array columns. Removing it.")
                # column 'multipeak' is in fact a Boolean
                if col == 'multipeak':
                    df[col] = df[col].astype(bool)
                else:
                    df = df.drop(columns=[col])
                array_colums.remove(col)

        # add component index (0 = SM1/SM2, 1=AF1, 2=AF2, etc.) and get time column to check whether the data is SIF
        column_to_test_size = [k for k, v in df.items() if all(x in k.lower() for x in ['time', 'tcb'])][0]

        if type(df.iloc[0][column_to_test_size]) is str:
            for col in array_colums:
                df[col] = df[col].apply(lambda x: list(eval(x)))

        component_index_len = len(df.iloc[0][column_to_test_size])
        df.loc[:, 'component_index'] = np.tile(np.arange(component_index_len), (len(df), 1)).tolist()

        logging.debug("Exploding the dataframe's CCD-level columns.")
        df_exploded = df.explode(array_colums + ['component_index']).reset_index(drop=True)
        return df_exploded

    def check_if_is_exploded(self, epoch_data):
        """Check whether the dataframe is already exploded.

        Parameters
        ----------
        input: pandas dataframe
            Dataframe to check whether it is exploded.

        Returns
        -------
        Boolean
            True or False if the dataframe is exploded or not.

        """
        check_if_list_like = is_list_like(epoch_data[self._time_column].iloc[0])

        if check_if_list_like:
            check_if_is_exploded = len(epoch_data[self._time_column].iloc[0])
            if check_if_is_exploded > 0:
                is_exploded = False
            else:
                is_exploded = True
        else:
            if isinstance(epoch_data[self._time_column].iloc[0], str):
                is_exploded = False
            else:
                is_exploded = True
        return is_exploded

    def filter_on_used_by_agis(self, scan_direction='al'):
        """Filter out rows containing observations not used by AGIS.

        Parameters
        ----------
        scan_direction: str, optional
            Scan direction to use, either 'al' (along scan) or 'ac' (across scan). Default: 'al'.

        Returns
        -------
        Pandas dataframe
            Dataframe filtered following column mask 'used_by_agis_al' or 'used_by_agis_ac'.

        """
        if scan_direction == 'al':
            filter_column = self._used_by_agis_al_column
        elif scan_direction == 'ac':
            filter_column = self._used_by_agis_ac_column

        index_keep = self._obj[f'{filter_column}'] == True
        logging.info(f'Filter on {filter_column}==True removed {len(self._obj) - index_keep.sum()} entries.')
        return self._obj[index_keep].reset_index(drop=True)

    def filter_on_multipeak(self):
        """Filter out rows where 'multipeak' column is True.

        Returns
        -------
        Pandas dataframe
            Dataframe without any entry with 'multipeak' column set to True.

        """
        index_keep = self._obj['multipeak'] == False
        logging.info(f'Filter on multipeak==False removed {len(self._obj) - index_keep.sum()} entries.')
        return self._obj[index_keep].reset_index(drop=True)

    def filter_on_ccdprocflags(self, flags_to_keep: np.ndarray):
        """Filter out rows where 'ccdprocflags' column values are in argument list 'flags_to_keep'.

        Parameters
        ----------
        flags_to_keep: narray
           List of values accepted in the 'ccdprocflags' column.

        Returns
        -------
        Pandas dataframe
            Dataframe without any entry with 'ccdprocflag' value in the list 'flags_to_keep'.

        """
        index_keep = self._obj['ccdProcFlags'].isin(flags_to_keep)
        logging.info(f'Filter on ccdProcFlags.isin({flags_to_keep}) removed {len(self._obj) - index_keep.sum()} entries.')
        return self._obj[index_keep].reset_index(drop=True)

    def filter_null_from_column(self, column):
        """Filter out rows where the specified 'column' contains nulls.

        Parameters
        ----------
        column: str
            Column name to be filtered.

        Returns
        -------
        Pandas dataframe
            Dataframe without any rows where 'column' contains null entries.

        """
        index_keep = self._obj[column].isnull() == False
        logging.info(f'Filter on column {column}!=null removed {len(self._obj) - index_keep.sum()} entries.')
        return self._obj[index_keep].reset_index(drop=True)

    def filter_out_skymapper(self):
        """Filter out rows corresponding to SkyMapper (SM) data.

        Returns
        -------
        Pandas dataframe
            Dataframe without any SkyMapper data.

        """
        index_keep = self._obj['component_index'] != 0
        logging.info(f'Filter on component_index!=0 removed {len(self._obj) - index_keep.sum()} SM entries.')
        return self._obj[index_keep].reset_index(drop=True)

    def filter_by_query(self, query):
        """Filter out the dataframe a string 'query'.

        Parameters
        ----------
        query: str
            String containing a query to filter data in the dataframe.

        Returns
        -------
        Pandas dataframe
            Dataframe filtered using the 'query' argument.

        """
        df_filtered = self._obj.query(query).reset_index(drop=True)
        logging.info(f'Filter using {query} removed {len(self._obj) - len(df_filtered)} entries.')
        return df_filtered

    def set_dtypes_for_exploded_columns(self, column_dtypes):
        """After exploding object-dtype columns their dtype needs to be set explicitly.

        Parameters
        ----------
        column_types: dict
            Dictionary of key/value wich corresponds to column/type.

        Returns
        -------
        Pandas dataframe
            Dataframe containing the columns/types as defined in 'column_types' argument.

        """
        return self._obj.astype(column_dtypes)

    def set_relative_time(self):
        """Set new columns for relative time in years and days, including the barycentric correction.

        Note
        ----
        GAIA-C3-TN-LU-LL-061 available at `Public DPAC documents <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`__.

        """
        logging.debug(f"Using reference epoch: {self._reference_epoch}")
        self._obj[self._relative_time_column_year] = TCB_REFERENCE_EPOCH.jyear + \
            self._obj[self._time_column] * (u.nanosecond.to(u.year)) \
            + self._obj[self._time_barycentric_correction_column] * (u.nanosecond.to(u.year))\
            - self._reference_epoch.jyear

        self._obj[self._relative_time_column_day] = self._obj[self._relative_time_column_year] * u.year.to(u.day)

    def set_scan_angle_derived_columns(self):
        """Add new columns to dataframe that correspond to scan-angle coefficients."""
        if 'sin_theta' in self._obj.columns:
            logging.debug('Scan angle columns already exist. Skipping.')
            return self._obj
        if self._relative_time_column_year is None:
            raise ValueError('Objecty attribute `_relative_time_column_year` is not set.')
        if self._scan_angle_column_deg is None:
            raise ValueError('Objecty attribute `_scan_angle_column_deg` is not set.')

        self._obj.loc[:, 'sin_theta'] = np.sin(np.deg2rad(self._obj.loc[:, self._scan_angle_column_deg]))
        self._obj.loc[:, 'cos_theta'] = np.cos(np.deg2rad(self._obj.loc[:, self._scan_angle_column_deg]))
        self._obj.loc[:, 'sin_theta_time'] = self._obj.loc[:, 'sin_theta'] * self._obj.loc[:, self._relative_time_column_year]
        self._obj.loc[:, 'cos_theta_time'] = self._obj.loc[:, 'cos_theta'] * self._obj.loc[:, self._relative_time_column_year]
        return self._obj

    def sort_by_column(self, column_name):
        """
        Sort the epoch data dataframe by the specified column.

        Parameters
        ----------
        column_name: str
            Sort dataframe by 'column_name' in ascending order.

        """
        self._obj.sort_values(column_name, inplace=True)
        self._obj.reset_index(inplace=True, drop=True)

    def get_design_equation_parameters(self, model='6p_constrained_colour', scan_direction='Al') -> OrderedDict:
        """Compute and return the design equation parameters.

        s : pandas dataframe with the lpcs of the source s

        t0 : the reference epoch in obmt ns (see obmt decodeur)

        withDeltat : if true use the lpc extra deltat time to reference epoch in year

        return : D,h the design matrix [alpha0,delta0,varpi,mualphastar,mudelta] and right hand side [w]

        Parameters
        ----------
        model: str
            Model corresponding to the number of astrometric parameters to solve. Default: '5p_single_source'.
        scan_direction: str, optional
            Scan direction to use, either 'al' (along scan) or 'ac' (across scan). Default: 'al'.

        Returns
        -------
        OrderedDict
            Ordered dictionary containing the astrometric parameters computed.

        """
        if scan_direction == 'Al':
            columns_5p_single_source = ['sin_theta', 'cos_theta', self._parallax_factor_column_al, 'sin_theta_time',
                                        'cos_theta_time']
            auxiliary_columns = [self._centroid_position_column_al, self._centroid_position_error_column_al,
                                 self._time_column, self._relative_time_column_year]

        result = OrderedDict()

        if model == '3p_single_source_without_offsets':
            columns_3p_single_source_without_offsets = [self._parallax_factor_column_al, 'sin_theta_time', 'cos_theta_time']
            columns_to_extract = columns_3p_single_source_without_offsets + auxiliary_columns
            lpc0 = self.set_scan_angle_derived_columns()[columns_to_extract].reset_index()
            selected_columns = columns_3p_single_source_without_offsets
            lpc = lpc0.copy()

        elif model == '5p_single_source':
            columns_to_extract = columns_5p_single_source + auxiliary_columns
            lpc0 = self.set_scan_angle_derived_columns()[columns_to_extract].reset_index()
            lpc = lpc0.copy()
            selected_columns = columns_5p_single_source

        elif model == '6p_constrained_colour':
            columns_to_extract = columns_5p_single_source + [self._colour_factor_column_al] + auxiliary_columns
            lpc0 = self.set_scan_angle_derived_columns()[columns_to_extract].reset_index()
            lpc = lpc0.copy()
            selected_columns = columns_5p_single_source + [self._colour_factor_column_al]

        result['design_matrix_coefficients'] = lpc[selected_columns].to_numpy()
        result['normal_matrix_column_names'] = selected_columns
        result['dependent_variable'] = lpc[self._centroid_position_column_al].values
        result['dependent_variable_error'] = lpc[self._centroid_position_error_column_al].values
        result['model'] = model
        result['timestamps'] = lpc[[self._time_column, self._relative_time_column_year]].reset_index(drop=True)
        return result


@pd.api.extensions.register_dataframe_accessor("epochastrometrycu9")
class EpochAstrometryDataFrameCu9(GaiaCentroidDataFrame):
    """Extension for Epoch Astrometric Data in CU9 format (internal use)."""

    _validation_columns = ['obsTimeBaryCorr', 'centroidPosAl', 'centroidPosErrorAl']

    _scan_angle_column = 'scanPosAngle'
    _scan_angle_column_deg = 'scanPosAngle'
    _parallax_factor_column_al = 'parallaxFactorAl'
    _colour_factor_column_al = 'colourFactorAl'
    _centroid_position_column_al = 'centroidPosAl'
    _centroid_position_error_column_al = 'centroidPosErrorAl'
    _time_column = 'obsTimeTcb'
    _time_barycentric_correction_column = 'obsTimeBaryCorr'
    _used_by_agis_al_column = 'usedByAgisAl'
    _used_by_agis_ac_column = 'usedByAgisAc'
    _blended_column = 'blended'
    _agis_source_excess_noise_column = 'agisSourceExcessNoise'
    _relative_time_column_year = 'relative_time_year'
    _relative_time_column_day = 'relative_time_day'
    _reference_epoch = DR4_REFERENCE_EPOCH


@pd.api.extensions.register_dataframe_accessor("epochastrometryarchive")
class EpochAstrometryDataFrameArchive(GaiaCentroidDataFrame):
    """Extension for Epoch Astrometric Data in public Gaia Archive format."""

    _validation_columns = ['obs_time_bary_corr', 'centroid_pos_al', 'centroid_pos_error_al']

    _scan_angle_column = 'scan_pos_angle'
    _scan_angle_column_deg = 'scan_pos_angle'
    _parallax_factor_column_al = 'parallax_factor_al'
    _colour_factor_column_al = 'colour_factor_al'
    _centroid_position_column_al = 'centroid_pos_al'
    _centroid_position_error_column_al = 'centroid_pos_error_al'
    _time_column = 'obs_time_tcb'
    _ipd_error_column_al = 'ipd_error_al'
    _time_barycentric_correction_column = 'obs_time_bary_corr'
    _used_by_agis_al_column = 'used_by_agis_al'
    _used_by_agis_ac_column = 'used_by_agis_ac'
    _blended_column = 'blended'
    _agis_source_excess_noise_column = 'agis_source_excess_noise'
    _relative_time_column_year = 'relative_time_year'
    _relative_time_column_day = 'relative_time_day'
    _reference_epoch = DR4_REFERENCE_EPOCH


class GaiaEpochAstrometry:
    """Class for Gaia Epoch Astrometric Data."""

    _data_is_exploded = None
    _time_column = None
    _scan_angle_column = None
    _centroid_position_column_al = None
    _centroid_position_error_column_al = None
    _ipd_error_column_al = None
    _agis_source_excess_noise_column = None
    _colour_factor_column_al = None

    n_filtered_ccd_transits = None

    def __init__(self, epoch_data: pd.DataFrame, source_id=None, explode=True, is_exploded=False):
        """Initialise GaiaEpochAstrometry instance.

        Parameters
        ----------
        epoch_data: pandas dataframe
            Dataframe containing epoch astrometry data.
        source_id: long int, optional
            Gaia identifier of the source being queried.
        explode: boolean, optional
            Whether the dataframe has to be exploded by device per row. Default: True
        is_exploded: boolean, optional
            Whether the dataframe is already exploded by device per row. Default: False

        """
        # Check whether the dataframe is already exploded by device per row
        is_exploded = GaiaCentroidDataFrame.check_if_is_exploded(self, epoch_data)
        self._data_is_exploded = is_exploded

        if explode and (self._data_is_exploded is False):
            self.epoch_data = GaiaCentroidDataFrame.explode_ccdlevel_columns(epoch_data.copy())
            # filter null entries from time column, otherwise data type cannot be set correctly
            self.epoch_data = self.epoch_data.gaiacentroid.filter_null_from_column(self._time_column)
            self.epoch_data = self.epoch_data.gaiacentroid.set_dtypes_for_exploded_columns(self._get_column_dtypes())
            self._data_is_exploded = True
        else:
            self.epoch_data = epoch_data

        if self._data_is_exploded:
            self.n_original_ccd_transits = len(self.epoch_data)

        if source_id is not None:
            self.source_id = source_id

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        """Extract data from dataframe and return astropy table.

        Parameters
        ----------
        df: pandas dataframe
            Data in pandas dataframe format.

        Returns
        -------
        Astropy table
            Astropy table containing the dataframe data.

        """
        return cls(df, **kwargs)

    @classmethod
    def _get_column_dtypes(cls):
        """Get the column datatypes of the columns that were exploded.

        Parameters
        ----------
        cls: astropy table
            Astropy table containing the dataframe data.

        Returns
        -------
        dict
            Dictionary of key/values wich corresponds to column/type.

        """
        column_dtypes = {cls._time_column: 'int64',  # Java Long
                         cls._scan_angle_column: 'float64',  # Java double
                         cls._centroid_position_column_al: 'float64',  # Java double
                         cls._centroid_position_error_column_al: 'float',  # Java float
                         cls._ipd_error_column_al: 'float',  # Java float
                         cls._colour_factor_column_al: 'float',  # Java float
                         cls._ccd_proc_flags_column: 'int16',
                         'component_index': 'int8',
                         }
        return column_dtypes


class GaiaEpochAstrometryCu9(GaiaEpochAstrometry):
    """Class for Gaia Epoch Astrometric Data in MDB format."""

    _source_id_field = 'sourceId'
    _transit_id_field = 'transitId'
    _time_barycentric_correction_column = 'obsTimeBaryCorr'
    _parallax_factor_column_al = 'parallaxFactorAl'

    _time_column = 'obsTimeTcb'
    _scan_angle_column = 'scanPosAngle'
    _centroid_position_column_al = 'centroidPosAl'
    _centroid_position_error_column_al = 'centroidPosErrorAl'
    _colour_factor_column_al = 'colourFactorAl'
    _ipd_error_column_al = 'ipdErrorAl'
    _ccd_proc_flags_column = 'ccdProcFlags'  # ccd_proc_flags
    _agis_source_excess_noise_column = 'agisSourceExcessNoise'

    @classmethod
    def filter_columns_for_computation(cls, df):
        """Dismiss the unneeded columns during the computation.

        Parameters
        ----------
        df: pandas dataframe
            Dataframe containing data.

        Returns
        -------
        Pandas dataframe
            Dataframe that containg only the columns needed for source update computation, the rest are dropped.

        """
        columns = ['sourceId', 'obsTimeBaryCorr', 'centroidPosAl', 'centroidPosErrorAl']
        columns += ['scanPosAngle', 'parallaxFactorAl', 'colourFactorAl', 'obsTimeTcb', 'usedByAgisAl']
        columns += ['agisSourceExcessNoise', 'ccdProcFlags', 'ipdErrorAl', 'nuEffUsedInAstrometry']
        return df[columns].copy()

    @classmethod
    def supdate(cls, df, sourceid, model=None, compute_excess_noise=False):
        """Run compute_source_parameters_like_dr4 on CU9 internal format.

        Parameters
        ----------
        df: pandas dataframe
            Dataframe containing data.
        source_id: long int
            Gaia identifier of the source.
        model: str, optional
            Model corresponding to the number of astrometric parameters solve. Default: 6p_constrained_colour.
        compute_excess_noise: boolean, optional
            Whether to fit the excess_noise of the observation or use computed by AGIS.

        Returns
        -------
        dict
            The astrometric parameters computed for the source.

        """
        if not model:
            model = '6p_constrained_colour'

        # Check whether the dataframe is already exploded by device per row
        is_exploded = GaiaCentroidDataFrame.check_if_is_exploded(GaiaEpochAstrometryCu9(df), df)

        df = cls.filter_columns_for_computation(df)
        ea_cu9_df = GaiaEpochAstrometryCu9(df, is_exploded=is_exploded).epoch_data
        ea_df = ea_cu9_df[ea_cu9_df[GaiaEpochAstrometryCu9(df, is_exploded=True)._source_id_field] == sourceid]
        ea_df.loc[:, 'colourFactorAl'] *= -1e3
        ea = GaiaSourceEpochAstrometryCu9.from_dataframe(ea_df, sourceid, is_exploded=True)

        if model == '6p_constrained_colour' and not compute_excess_noise:
            supdate = ea.compute_source_parameters_like_dr4()
        else:
            supdate = ea.compute_source_parameters(model, compute_excess_noise)
        return supdate


class GaiaEpochAstrometryArchive(GaiaEpochAstrometry):
    """Class for Gaia Epoch Astrometric Data in Gaia Archive format."""

    _source_id_field = 'source_id'
    _transit_id_field = 'transit_id'
    _time_column = 'obs_time_tcb'
    _time_barycentric_correction_column = 'obs_time_bary_corr'
    _parallax_factor_column_al = 'parallax_factor_al'
    _colour_factor_column_al = 'colour_factor_al'
    _scan_angle_column = 'scan_pos_angle'
    _centroid_position_column_al = 'centroid_pos_al'
    _centroid_position_error_column_al = 'centroid_pos_error_al'
    _ipd_error_column_al = 'ipd_error_al'
    _ccd_proc_flags_column = 'ccd_proc_flags'

    @classmethod
    def astropy_table_to_df(cls, df):
        """Convert series in astropy table in flat df.

        Parameters
        ----------
        df: Pandas dataframe
            Dataframe containing data.

        Returns
        -------
        Pandas dataframe
            Dataframe with series from astropy flatten.

        """
        for col in list(df.columns):
            if 'MaskedArray' in str(type(df.iloc[0][col])):
                df[col] = df[col].apply(lambda x: x.data)
        return df

    @classmethod
    def archive_to_cu9(cls, df):
        """Convert dataframe from Gaia Archive format to CU9 format.

        Parameters
        ----------
        df: Pandas dataframe
            Dataframe containing data in Gaia Archive format.

        Returns
        -------
        Pandas dataframe
            Dataframe containing data in CU9 format.

        """
        cols_archive_to_cu9 = {
            'source_id': 'sourceId',
            'obs_time_bary_corr': 'obsTimeBaryCorr',
            'centroid_pos_al': 'centroidPosAl',
            'centroid_pos_error_al': 'centroidPosErrorAl',
            'scan_pos_angle': 'scanPosAngle',
            'parallax_factor_al': 'parallaxFactorAl',
            'colour_factor_al': 'colourFactorAl',
            'obs_time_tcb': 'obsTimeTcb',
            'used_by_agis_al': 'usedByAgisAl',
            'agis_source_excess_noise': 'agisSourceExcessNoise',
            'ccd_proc_flags': 'ccdProcFlags',
            'ipd_error_al': 'ipdErrorAl',
            'nu_eff_used_in_astrometry': 'nuEffUsedInAstrometry',
        }

        df = cls.astropy_table_to_df(df)
        epoch_astrometry_cu9 = df[list(cols_archive_to_cu9.keys())].copy()
        epoch_astrometry_cu9.rename(columns=cols_archive_to_cu9, inplace=True)
        return epoch_astrometry_cu9

    @classmethod
    def supdate(cls, df, sourceid, model=None, compute_excess_noise=False):
        """Run compute_source_parameters_like_dr4 from Archive data.

        Parameters
        ----------
        df: Pandas dataframe
            Dataframe containing data in Gaia Archive format.
        source_id: long int
            Gaia identifier of the source.
        model: str, optional
            Model corresponding to the number of astrometric parameters to solve. Default: '6p_constrained_colour'.
        compute_excess_noise: boolean, optional
            Whether to fit the excess_noise of the observation or use computed by AGIS.

        Returns
        -------
        dict
            The astrometric parameters computed for the source.

        """
        if not model:
            model = '6p_constrained_colour'

        d = cls.archive_to_cu9(df)
        supdate = GaiaEpochAstrometryCu9.supdate(d, sourceid, model, compute_excess_noise)
        return supdate


class GaiaSourceEpochAstrometry(GaiaEpochAstrometry):
    """Class for Gaia Epoch Astrometric Data corresponding to an individual source."""

    @classmethod
    def from_dataframe(cls, df, source_id, **kwargs):
        """Extract data from dataframe and return astropy table for a specific source_id.

        Parameters
        ----------
        df: Pandas dataframe
            Data in pandas format.
        source_id: long int
            Gaia identifier of the source.

        Returns
        -------
        Astropy table
            Data in Astropy table format for a specific source_id.

        """
        if df[cls._source_id_field].unique() != source_id:
            raise ValueError('Input dataframe has non-unique or non-matching source_id.')
        return cls(df, source_id=source_id, **kwargs)

    def compute_source_update(self, model='5p_single_source'):
        """Execute source update for model '5p_single_source'.

        Parameters
        ----------
        model: str
            Model corresponding to the number of astrometric parameters to solve. Default: '5p_single_source'.

        Returns
        -------
        dict
            The astrometric parameters computed for the source.

        """
        selected_source_id = self.source_id
        result_parameters = {'sourceId': selected_source_id}
        try:
            design_parameters = self.get_design_parameters(model=model)
            design_equation = DesignEquation(design_parameters)
            results = design_equation.solve()
            result_parameters['results'] = results
            result_parameters['success'] = 1
        except (LinAlgError, AttributeError) as e:
            result_parameters['success'] = 0
            logging.warning(e)

        return result_parameters

    def compute_perspective_acceleration(self, convergence_threshold=1e-9):
        """Compute a source update with perspective acceleration and return results.

        Parameters
        ----------
        convergence_threshold: float, optional
            Convergence threshold. Default: 1e-9.

        Returns
        -------
        dict
            Dictionary containing the results of the computation.

        """
        selected_source_id = self.source_id
        result_parameters = {'sourceId': selected_source_id}

        initial_parameters = np.array([0., 0., 0., 0., 0.])
        results_5p = self.compute_source_update_iteratively(initial_parameters, fit_perspective_acceleration=False)
        parameters_5p = results_5p['parameters']

        initial_parameters_perspective_acceleration = np.hstack((parameters_5p, [0]))
        results = self.compute_source_update_iteratively(initial_parameters_perspective_acceleration,
                                                         fit_perspective_acceleration=True,
                                                         convergence_threshold=convergence_threshold)
        result_parameters['results'] = results
        result_parameters['success'] = 1
        return result_parameters

    def compute_source_update_iteratively(self, initial_parameters, fit_perspective_acceleration=False,
                                          n_iterations_max=10, solver='least_squares', convergence_threshold=1e-9):
        """Return iterative source update which allows us to account for perspective acceleration.

        Parameters
        ----------
        initial_parameters: narray
            Initial parameters to start the iteration.
        fit_perspective_acceleration: boolean, optional
            Whether to fit for perspective acceleration. Default: False
        n_iterations_max: int, optional
            Number of maximum iterations to perform. Default: 10.
        solver: str, optional
            Solver to be used. Default: least_squares.
        convergence_threshold: float, optional
            Maximum convergence threshold. Default: 1e-9.

        Returns
        -------
        dict
            The parameters of the solution.

        Note
        ----
        Equations were implemented following GAIA-C3-TN-LU-LL-061 (Appendix E) available at `Public DPAC documents <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`__.

        """
        for col in ['relative_time_year', 'sin_theta']:
            if col not in self.epoch_data.columns:
                logging.error(f"Method requires column {col} in epoch data. \
                Try executing set_relative_time() or set_scan_angle_derived_columns() first.")

        lpc = self.epoch_data.copy()
        lpc['fr'] = -lpc[self._time_barycentric_correction_column] / 1e9 / 499.004783836156

        results = []
        solution_parameters = initial_parameters
        for iteration_number in np.arange(n_iterations_max):

            if iteration_number == 0:
                pass
            else:
                results_previous_interation = copy.deepcopy(results)
                solution_parameters += results_previous_interation['parameters']

            # set initial parameters
            if fit_perspective_acceleration is False:
                mu_r_0 = 0
                model = '5p_single_source'
            else:
                mu_r_0 = solution_parameters[-1]
                model = '6p_perspective_acceleration'

            varpi_0 = solution_parameters[2]

            d_factor = 1 + (lpc['relative_time_year'] * mu_r_0 + lpc['fr'] * varpi_0) * mas_to_rad

            lpc['sin_theta/d_factor'] = lpc['sin_theta'] / d_factor
            lpc['cos_theta/d_factor'] = lpc['cos_theta'] / d_factor
            lpc['sin_theta_time/d_factor'] = lpc['sin_theta_time'] / d_factor
            lpc['cos_theta_time/d_factor'] = lpc['cos_theta_time'] / d_factor

            selected_columns_for_w_calculated = ['sin_theta', 'cos_theta', self._parallax_factor_column_al,
                                                 'sin_theta_time', 'cos_theta_time']
            w_calculated = (lpc[selected_columns_for_w_calculated].to_numpy() @ solution_parameters[0:5]) / d_factor

            lpc['varpiFactorAl'] = (lpc[self._parallax_factor_column_al] - w_calculated * lpc['fr'] * mas_to_rad) / d_factor

            if fit_perspective_acceleration:
                lpc['perspAccelFactor'] = -1 * w_calculated * lpc['relative_time_year'] * mas_to_rad / d_factor
                selected_columns_for_matrix = ['sin_theta/d_factor', 'cos_theta/d_factor',
                                               'varpiFactorAl', 'sin_theta_time/d_factor',
                                               'cos_theta_time/d_factor', 'perspAccelFactor']
            else:
                selected_columns_for_matrix = ['sin_theta/d_factor', 'cos_theta/d_factor',
                                               'varpiFactorAl', 'sin_theta_time/d_factor',
                                               'cos_theta_time/d_factor']

            design_parameters = OrderedDict()
            design_parameters['design_matrix_coefficients'] = lpc[selected_columns_for_matrix].to_numpy()
            design_parameters['dependent_variable'] = lpc[self._centroid_position_column_al].values - w_calculated.values
            design_parameters['dependent_variable_error'] = lpc[self._centroid_position_error_column_al].values
            design_parameters['model'] = model

            design_equation = DesignEquation(design_parameters)
            results = design_equation.solve(solver=solver)

            for i in range(len(results['parameters'])):
                logging.debug(f"{solution_parameters[i]:+15.6f} +/- {results['parameters_normalised_uncertainty'][i]:<+15.6f}")

            logging.debug(f"iteration {iteration_number}: maximum update {np.max(results['parameters'])} \
            minimum update {np.min(results['parameters'])}")
            if (np.max(np.abs(results['parameters'])) < convergence_threshold):
                logging.info(f"iterations converged after {iteration_number} iterations.")
                results_previous_interation['parameters'] = solution_parameters
                return results_previous_interation

        logging.warning(f"iterations did not converge after {iteration_number} iterations.")
        logging.warning(f"iteration {iteration_number}: maximum update {np.max(results['parameters'])} \
        minimum update {np.min(results['parameters'])}")


class GaiaSourceEpochAstrometryCu9(GaiaEpochAstrometryCu9, GaiaSourceEpochAstrometry):
    """Class for Gaia Epoch Astrometric Data in CU9 format corresponding to an individual source."""

    def __str__(self):
        """Return string describing the instance."""
        return 'GaiaSourceEpochAstrometryCu9 for source_id {} with {} CCD \
        transits'.format(self.source_id, len(self.epoch_data))

    def get_design_parameters(self, filter_on_used_by_agis=False, **kwargs):
        """Compute the design parameters using class-specific column names.

        Parameters
        ----------
        filter_on_used_by_agis: boolean, optional
            Select whether to follow the selection strategy as AGIS for the parameters computation. Default: False.

        """
        if filter_on_used_by_agis:
            self.epoch_data = self.epoch_data.epochastrometrycu9.filter_on_used_by_agis()
        self.epoch_data.epochastrometrycu9.set_relative_time()
        design_parameters = self.epoch_data.epochastrometrycu9.get_design_equation_parameters(**kwargs)
        return design_parameters

    def compute_source_parameters(self, model=None, compute_excess_noise=False):
        """Compute the astrometric source paramaters.

        Parameters
        ----------
        model: str, optional
            Model corresponding to the number of astrometric parameters to solve. Default: 6p_constrained_colour.
        compute_excess_noise: boolean, optional. Default: False.
            Whether to fit the excess noise or to use the provided with the epoch astrometry data.

        Returns
        -------
        dict
            The astrometric parameters computed for the specified source.

        """
        if not model:
            model = '6p_constrained_colour'

        # apply the appropriate filters
        self.epoch_data = self.epoch_data.epochastrometrycu9.filter_on_used_by_agis()
        self.epoch_data = self.epoch_data.gaiacentroid.filter_null_from_column(self.epoch_data.epochastrometrycu9.
                                                                               _time_barycentric_correction_column)
        self.epoch_data = self.epoch_data.gaiacentroid.filter_null_from_column(self.epoch_data.epochastrometrycu9.
                                                                               _scan_angle_column)
        # compute time stamps
        self.epoch_data.epochastrometrycu9.set_relative_time()
        logging.info(self)

        # generate design parameters
        design_parameters = self.epoch_data.epochastrometrycu9.get_design_equation_parameters(model=model)
        n_linear_parameters = design_parameters['design_matrix_coefficients'].shape[1]

        if model == '6p_constrained_colour':
            assert n_linear_parameters == 6, "Number of linear parameters should be 6."

            # set prior on pseudolour offset
            prior_strength_nm = 0.085e-3  # this is in units of 1/nm
            design_parameters['gaussian_priors'] = np.array([None] * (n_linear_parameters - 1) + [prior_strength_nm])

        if model == '5p_single_source':
            assert n_linear_parameters == 5, "Number of linear parameters should be 5."

        # compute design equation
        design_equation = DesignEquation(design_parameters)

        if not compute_excess_noise:
            # set excess source noise (this originates from AGIS)
            excess_source_noise_mas = self.epoch_data.iloc[0][self._agis_source_excess_noise_column]
            excess_source_variance = excess_source_noise_mas**2
            total_variance = design_equation.observation_variances + excess_source_variance
        else:
            total_variance = None

        # compute source parameters but solving equations
        results = design_equation.solve(solver='agis', total_variance=total_variance)
        return results

    def compute_source_parameters_like_dr4(self):
        """Compute the astrometric source paramaters with a DR4-like configuration.

        This will generate the best-possible reproduction of the DR4 astrometric parameters.

        Returns
        -------
        dict
            The astrometric parameters computed for the specified source.

        """
        model = '6p_constrained_colour'
        compute_excess_noise = False

        results = self.compute_source_parameters(model, compute_excess_noise)
        return results


class GaiaSourceEpochAstrometryArchive(GaiaEpochAstrometryArchive, GaiaSourceEpochAstrometry):
    """Class for Gaia Epoch Astrometric Data in the Gaia Archive format corresponding to an individual source."""

    def __str__(self):
        """Return a string describing the instance.

        Constructor of GaiaSourceEpochAstrometryArchive.
        """
        return 'GaiaSourceEpochAstrometryArchive for source_id {} with {} \
        CCD transits'.format(self.source_id, len(self.epoch_data))

    def get_design_parameters(self, filter_on_used_by_agis=True, **kwargs):
        """Compute the design parameters using class-specific column names.

        Parameters
        ----------
        filter_on_used_by_agis: boolean, optional
            Select whether to follow the selection strategy as AGIS for the parameters computation.
            Default: True

        Returns
        -------
        design parameters: OrderedDict
            Dictionary containing the design_matrix_coefficients and auxiliary data.

        """
        if filter_on_used_by_agis:
            self.epoch_data = self.epoch_data.epochastrometryarchive.filter_on_used_by_agis()
        self.epoch_data.epochastrometryarchive.set_relative_time()
        design_parameters = self.epoch_data.epochastrometryarchive.get_design_equation_parameters(**kwargs)
        return design_parameters

    @classmethod
    def from_gacs_datalink(cls, source_id, format='votable', gaia_data_server='https://gea.esac.esa.int/',
                           credentials_file=None, data_release='Gaia DR4_INT4', data_structure='RAW',
                           retrieval_type='EPOCH_ASTROMETRY'):
        """Query Gaia Archive to retrive EpochAstrometry data of a specific object.

        Parameters
        ----------
        source_id: long int
            Gaia identifier of the source being queried.
        format: str, optional
            Format of the data to be retrieved. It can be VOTable, CSV, ECSV, FITS or parquet
        gaia_data_server: str, optional
            URL of the Archive data server. Default is https://gea.esac.esa.int/
        credentials_file: str, optional
            Local path of the txt file containing the user Archive credentials.

        Returns
        -------
        Astropy table
            Table that contains the epoch astrometry data requested to the Gaia Archive.

        """
        if (gaia_data_server != 'https://gea.esac.esa.int/') or (credentials_file is not None):
            gaia = GaiaClass(gaia_tap_server=gaia_data_server, gaia_data_server=gaia_data_server)
            gaia.login(credentials_file=credentials_file)
        else:
            gaia = Gaia()

        ea_data = gaia.load_data(ids=[source_id], data_release=data_release,
                                 retrieval_type=retrieval_type, format=format,
                                 data_structure=data_structure)
        if format == 'votable':
            epoch_astro_df = ea_data[f"{retrieval_type}-{data_release} {source_id}.xml"][0].to_table().to_pandas()
            epoch_astro_df = GaiaEpochAstrometryArchive.astropy_table_to_df(epoch_astro_df)
        if format == 'csv':
            epoch_astro_df = ea_data[f"{retrieval_type}-{data_release} {source_id}.csv"][0].to_pandas()

        return cls.from_dataframe(epoch_astro_df, source_id)
