#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Compute astrometric source parameters from Gaia epoch astrometry."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gaiasupdate")
except PackageNotFoundError:
    pass
