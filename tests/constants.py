#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Module that defines testing constants."""

import os

_THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

TEST_DATA_ROOT = os.path.join(_THIS_DIRECTORY, 'test_data')
