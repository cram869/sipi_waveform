#!/usr/bin/env python3
"""
Utility functions used by multiple files within this package

original author: Michael Cracraft (cracraft@rose-hulman.edu)
"""
############################################################################
import time

def print_timing(func):
    """print_timing is a decorator function to return the execution time 
    of func."""
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        t2 = time.time()

        dt = t2 - t1
        if dt > 0.09:
            print('%s took %0.6f s' % (func.__name__, (t2-t1)))
        else:
            print('%s took %0.3f ms' % (func.__name__, (t2-t1)*1000.0))
        return res
    return wrapper