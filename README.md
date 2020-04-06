
# Exploration of tools, data from Aravkin et al 2020 (forthcoming)

This is highly experimental and does not match official projections

Warning
-------

This is NOT the official repository for the IHME COVID model. See their repo at https://github.com/ihmeuw-msca/CurveFit

# Usage

Execute the python script ``python main.py`` or run the ipython notebook using Binder:


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/delgadom/CurveFit/HiroakiMachida?urlpath=lab%2Ftree%2Fmain.ipynb)

## Install

Clone or download the repository and then do:
```buildoutcfg
make install
```

If you want to install somewhere other than the defualt for your system:
```
make install prefix=install_prefix_directory
```
Required packages:
* `numpy`,
* `scipy`,
* `pandas`. 
