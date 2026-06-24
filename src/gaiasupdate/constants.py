#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Module that defines central constants and variables used in this package.

Attributes
----------
TCB_REFERENCE_EPOCH: Astropy time
    Barycentric Coordinate Time reference epoch 2010-01-01T00:00:00 (TCB).
DR4_REFERENCE_EPOCH: Astropy time
    Gaia Data Release 4 reference epoch 2017.5 julian year (TCB).

"""

__author__ = "Johannes Sahlmann"

from astropy.time import Time

TCB_REFERENCE_EPOCH = Time('2010-01-01T00:00:00', format='isot', scale='tcb')

DR4_REFERENCE_EPOCH = Time('2017.5', format='jyear', scale='tcb')
