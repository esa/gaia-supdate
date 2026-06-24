..   Copyright (c) European Space Agency, 2026.
..
..   This file is subject to the terms and conditions defined in file "LICENSE.txt", which
..   is part of this source code package. No part of the package, including
..   this file, may be copied, modified, propagated, or distributed except according to
..   the terms contained in the file "LICENSE.txt".



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   installation

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Usage

   tools
   supdate_output_params


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Package documentation

   modules


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Other
   :titlesonly:

   citation_license



.. image:: _static/gaia_mission_logo.png
   :width: 200px
   :align: right
   :alt: Gaia mission logo


GaiaSupdate
===========

The main objective of the `gaiasupdate` package is the computation of the astrometric source parameters from Gaia epoch astrometry data.

The fourth Gaia data release (Gaia DR4) will make public the astrometric source parameters, e.g. the parallax and proper motions, alongside with timeseries of individual positional measurements, i.e. the "epoch astrometry" which will be available via DataLink or bulk download.

The `gaiasupdate` software package allows users to compute the former on the basis of the latter, to the extent possible.


Features
--------

* Reproduce per source the astrometric parameters in the *gaiadr4.all_source_astrometry* Gaia Archive table using epoch astrometry data.

* An interface to the Gaia Archive via Astroquery for epoch astrometry data retrieval.

Note that the epoch data of all sources will not be publicly available until the Gaia DR4 release. 

Epoch astrometry of a small sample sources was released in advance of Gaia DR4, alongside this package.







