#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Module with basic utilities."""

__author__ = "Johannes Sahlmann"

from collections import OrderedDict
import numpy as np


def fov_from_transit_id(transit_id: int) -> int:
    """Extract the FoV information from a transit_id.

    Parameters
    ----------
    transit_id : int
        Transit identifier.

    Returns
    -------
    int
        Gaia field of view (1 or 2).

    Note
    ----
    See GAIA-C3-TN-UB-JP-011, page 9 available at `Public DPAC documents <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`__.

    """
    fov = np.byte(transit_id >> 15) & 0x03
    return fov


# CCD processing flags information from archive documentation (MDB/CU3/ID/AstroElementary MDB trunk data model)
ccd_proc_flags_decode_dict = OrderedDict({0x000F: 'IPD problem',  # ==0->success, !=0->failure
                                          0x0010: 'IPD non nominal (reduced window area)',
                                          0x0020: 'IPD not available',
                                          0x0040: 'Cosmetic issue',
                                          0x0080: 'Cosmic removed',
                                          0x0100: 'Saturation removed',
                                          0x0200: 'Part of window discarded',
                                          0x0400: 'No window',
                                          0x0800: 'Non-success InitialCentroid',
                                          0x1000: 'Odd Background found',
                                          0x2000: 'fall-back to lower Bias Mitigation mode',
                                          0x4000: 'non-target source removed',
                                          0x8000: 'LSF/PSF observation/source parameters clamped'
                                          })

ccd_proc_flags_decode_keys = np.array(list(ccd_proc_flags_decode_dict.keys()))
ccd_proc_flags_decode_values = np.array(list(ccd_proc_flags_decode_dict.values()))


def decode_ccd_proc_flag_to_description(flag: int) -> np.ndarray:
    """Return array of strings corresponding to the CCD processing flag.

    Parameters
    ----------
    flag: int
        Processing flag.

    Returns
    -------
    narray
        Array containing CCD processing flags decoded.

    """
    matched_keys = np.bitwise_and(flag, ccd_proc_flags_decode_keys)
    matched_index = np.where(matched_keys != 0)
    description_array = ccd_proc_flags_decode_values[matched_index]
    return description_array


transit_acquisition_flags_decode_dict = OrderedDict({0x2000: 'Active Centring Inhibited',
                                                     0x1000: 'Trimmed AF',
                                                     0x0180: 'xpOffset',
                                                     0x0040: '2D AF windows',
                                                     0x0020: 'Truncated or trimmed AF',
                                                     0x0010: 'Truncated AF with full (nominal, non-trimmed) geometry',
                                                     })

transit_acquisition_flags_decode_keys = np.array(list(transit_acquisition_flags_decode_dict.keys()))
transit_acquisition_flags_decode_values = np.array(list(transit_acquisition_flags_decode_dict.values()))


def decode_transit_acquisition_flag_to_description(flag: int) -> np.ndarray:
    """Return array of strings corresponding to the CCD processing flag.

    Parameters
    ----------
    flag: int
        Processing flag.

    Returns
    -------
    narray
        Array containing the transit aquisition flags decoded.

    """
    matched_keys = np.bitwise_and(flag, transit_acquisition_flags_decode_keys)
    matched_index = np.where(matched_keys != 0)
    description_array = transit_acquisition_flags_decode_values[matched_index]
    return description_array
