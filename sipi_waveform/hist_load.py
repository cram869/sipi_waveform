#!/usr/bin/env python3

"""
Package for loading histogram save files from Tektronix oscilloscopes.

original author: Michael Cracraft (macracra@us.ibm.com)
"""
from numpy import array as np_array

def tek_histogram_load(filename):
    """
    The Tektronix histogram format is simply a CSV file with voltage bins in the
    first column and the hit counts in the second column.  No column headers are
    given.
    """
    with open(filename, 'r') as fobj:
        rows = [l_.strip().split(',') for l_ in fobj.read().strip().splitlines()]
        voltage_bins = np_array([float(r_[0]) for r_ in rows])
        hit_counts = np_array([int(r_[1]) for r_ in rows])
        return voltage_bins, hit_counts
