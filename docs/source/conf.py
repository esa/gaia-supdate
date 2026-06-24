#   Copyright (c) European Space Agency, 2026.
#
#   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file "LICENSE.txt".

"""Configuration file for the Sphinx documentation builder."""

import sys
import os
import gaiasupdate
from sphinx_pyproject import SphinxConfig
from importlib.metadata import version, PackageNotFoundError
import tomli as tomllib

# Path to the source code
sys.path.insert(0, os.path.abspath('../../src/'))

# Get package release version
try:
    release = version("gaiasupdate")
except PackageNotFoundError:
    release = "0.0.0+unknown"

# Get short version number for the documentation
try:
    version = gaiasupdate.__version__
except PackageNotFoundError:
    version = ".".join(release.split(".")[:2])

# Location of the toml file in the package
toml_path = "../../pyproject.toml"

config = SphinxConfig(toml_path, globalns=globals(),
                      config_overrides={"version": version, "release": release})

# Get toml file configuration
try:
    with open(toml_path, mode="rb") as fp:
        toml_config = tomllib.load(fp)
        # Re-setting some variables since there are problems getting them
        project = toml_config.get("project", {}).get("name", "My Project")
        html_theme_options = {}
        theme_options_from_toml = toml_config.get("tool", {}).get("sphinx", {}).get("html_theme_options", {})
        html_theme_options.update(theme_options_from_toml)
except FileNotFoundError:
    print(f"Error: TOML configuration file not found at {toml_path}", file=sys.stderr)
    toml_config = {}
