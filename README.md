#RS_SiVM - Remote Sensing with Simplex Volume Maximization

#Description
------------
This project contains a Python class for easy analysis of hyperspectral imagery time series with simplex volume maximisation (SiVM). SiVM is a fast matrix factorization technique for pattern recognition in high-dimensional input data. It can be used to extract characteristic extremes (archetypes) from hyperspectral remote sensing data, allowing the automatic detection of vegetation stress and other information.

The Factorator class currently includes methods for
* preprocessing
* resampling
* factorization
* visualization
of imagery time series.
The project also includes R code for the statistical evaluation of resulting vegetation stress products.

For information about usage of the Factorator class, take a look at the demo file.

Please note that the code has only been tested with Python 3.7.3 on Ubuntu 20.04.2 LTS. Error handling is almost non-existent. So expect bugs and problems in scalability when using code from this project.

Data for testing is available at ...

#Requirements
------------

Python code requires the following modules:

 * pymf 0.3 (not available on conda/pypi, use included version, source: https://github.com/cthurau/pymf)
 * fmch (not available on conda/pypi, use included version)
 * numpy 1.18.5
 * scipy 1.4.1
 * math
 * colorsys
 * datetime
 * pickle 
 * pandas 0.25.2
 * geopandas 0.6.1
 * matplotlib 3.1.1
 * spectral 0.20 (only available on pypi)
 * pathlib 2.3.5
 * json 2.0.9

recommended:
 * HSI2RGB (https://github.com/JakobSig/HSI2RGB)

The Python Matrix Factorization Module (https://github.com/cthurau/pymf) was ported to Python 3 before usage and is therefore re-distributed in this repository for reproducible results.

R code requires the following packages:

 * corrplot 0.84
 * ellipse 0.4.2
 * Hmisc 4.3-0
 * caret 6.0-84
 * mctest 1.3.1
 * betareg 3.1-3
 * gamboostLSS 2.0-1.1
 * ggplot2 3.3.2

#Installation
------------

Installation has been tested using conda 4.8.3 and Python 3.7.3 on Ubuntu 20.04.2 LTS.

1. If you install the dependencies manually, be sure to use conda-forge channel for all packages that are available there. Alternatively, you can try to import the whole environment into conda using the included YAML file.

2. The included dependencies "pymf" and "fmch" must be installed manually. Move them into your python environment's "site-packages" folder or another location that is included in your PYTHONPATH.


#factorator.py license: BSD 3-Clause
------------

Copyright 2021 Floris Hermanns

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#PyMF license: BSD 3-Clause
------------

Copyright 2014 Christian Thurau

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
