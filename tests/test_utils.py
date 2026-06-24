#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Tests for the gaiasupdate utils."""

__author__ = "Johannes Sahlmann"

import numpy as np
from numpy.testing import assert_array_equal
from gaiasupdate.utils import decode_ccd_proc_flag_to_description


def test_decode_ccd_proc_flag():
    """Test flag decoder."""
    flag = 9024
    result = decode_ccd_proc_flag_to_description(flag)
    expected_result = np.array(['Cosmetic issue', 'Saturation removed',
                                'Part of window discarded', 'fall-back to lower Bias Mitigation mode'])

    assert_array_equal(result, expected_result)
