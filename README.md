# sipi_waveform
`sipi_waveform` is a collection of Python routines for waveform processing.
Some routines are more mature than others.
User beware. 

# Installation
The package can be installed with pip or manually placed in the PYTHONPATH.
Assume the repository has been cloned into the path ~/py/sipi_waveform.

`python -m pip install ~/py/sipi_waveform`

The package can also be installed directly from the Github repo.  

`python -m pip install git+https://github.com/cram869/sipi_waveform.git`

# Updating
Updating the package may be completed by adding the "-U" switch to the pip command.
For example,
`python -m pip install -U git+https://github.com/cram869/sipi_waveform.git`

# Testing
The `test` folder includes a Jupyter notebook and example Tektronix wfm file (binary format) to be used to verify a few of the package functions.
The current tests are no where near exhaustive.
