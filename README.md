<!--
#  Copyright (c) European Space Agency, 2026.
#
#  This file is subject to the terms and conditions defined in file "LICENSE.txt", which
#  is part of this source code package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file "LICENSE.txt".
-->

[![License: ESA permissive](https://img.shields.io/badge/ESA%20Public%20License-Permissive-blue.svg)](https://github.com/esa/gaia-supdate/blob/main/LICENCE.txt)
[![pipeline status](https://github.com/esa/gaia-supdate/actions/workflows/ci_run_tests.yml/badge.svg?branch=prerelease)](https://github.com/esa/gaia-supdate/actions/workflows/ci_run_tests.yml)
[![docs](https://github.com/esa/gaia-supdate/documentation.svg)](https://esa.github.io/gaia-supdate)

<img src="docs/source/_static/gaia_mission_logo.png" alt="drawing" width="200"/>

# gaiasupdate

The main objective of the `gaiasupdate` package is the computation of the astrometric source parameters, e.g. the parallax and proper motions, from Gaia astrometric timeseries data, also known as Gaia "epoch astrometry".

The fourth Gaia data release ([Gaia DR4](https://www.cosmos.esa.int/web/gaia/data-release-4)) will publish the astrometric source parameters alongside with timeseries of individual positional measurements, i.e. the epoch astrometry which will be available via DataLink or bulk download. The `gaiasupdate` software package allows users to compute the former on the basis of the latter, to the extent possible.

### Features

* Compute astrometric source parameters on the basis of epoch astrometry data. 

Note that the epoch data of all sources will not be publicly available until the Gaia DR4 release. However, epoch astrometry of a small sample of sources was pre-released in advance of Gaia DR4.

### Naming

`gaiasupdate` is a shorthand for "Gaia source update", where "source update" refers to the determination of a source's astrometric parameters. This namenclature originates in the corresponding step of the astrometric global iterative solution, see [Lindegren et al. 2012](https://www.aanda.org/articles/aa/full_html/2012/02/aa17905-11/aa17905-11.html) (Sect. 4.1).  

### Installation

This package requires **Python >= 3.9** and the following dependencies:

+ numpy
+ scipy
+ pandas
+ astropy
+ pyarrow
+ astroquery

See [requirements.txt](requirements.txt) file for more details.

#### From source
```commandline
git clone https://github.com/esa/gaia-supdate
cd gaiasupdate
pip install -e .
```

#### From PyPI

```commandline
pip install gaiasupdate
```

#### Using conda environment

```commandline
conda create --name gaiasupdate-env --yes python=3.12 -r requirements.txt
conda activate gaiasupdate-env
```

### Usage examples

A notebook to show how to use `gaiasupdate` to compute source astrometric parameters from epoch astrometry data can be found at https://github.com/esa/gaia-jupyter-notebooks/tree/main/data-release-4-tutorials

There are also tests included in this package to show:
- how to produce the astrometric parameters of a source using epoch data: [Test source update](gaiasupdate/tests/test_constrained_colour_update.py)

### Documentation

The package documentation can be found at https://esa.github.io/gaia-supdate.


### Citation

If you make use of `gaiasupdate` in your research or otherwise, we would appreciate a citation of ZENODO BADGE and an acknowledgement along the lines of: 

"This work made use of the `gaiasupdate` package that is described at https://www.cosmos.esa.int/web/gaia/gaia-source-update.

More information on how to acknowledge Gaia resources is available at [Gaia credits and acknowledgements](https://www.cosmos.esa.int/web/gaia-users/credits).


### Changelog

* `gaiasupdate v0.0.1` - 2026/06/XX

    This pre-release version ahead of Gaia DR4 shows how to use Gaia epoch astrometry data.


### Acknowledgements

This code was written by A. Delgado and J. Sahlmann with contributions by A. Bombrun, L. Lindegren, and others in the Gaia collaboration.

### License

The details of the license of this package can be found in the file [LICENSE.txt](LICENSE.txt).

[European Space Agency Public License (ESA-PL) Permissive (Type 3) – v2.4](https://essr.esa.int/license/european-space-agency-public-license-v2-4-permissive-type-3).


