#!/usr/bin/env python3
import matplotlib.pyplot as plt
from numpy import mean as np_mean
from os.path import join as os_path_join

from sipi_waveform.wfm import tek_mdo_csvread
from sipi_waveform.tracemath import plotEye

###############################################################################
default_save_formats = ['png', 'pdf']
def savefigure(fig, filename, path='plots', **kwargs):
    pathfilename = os_path_join(path, filename)
    formats = kwargs.get('formats', default_save_formats)
    transparent = kwargs.get('transparent', False)
    for format_i in formats:
        fig.savefig(pathfilename + '.' + format_i,
                    transparent=transparent)
###############################################################################

fn = '10MHz random.csv'
trace_d = tek_mdo_csvread(fn)

# Dictionary output from tek_mdo_csvread will include multiple series
# with TIME and other labels.  In this case CH2 was the source of the 
# voltage information.
t_ = trace_d['TIME']
v_ = trace_d['CH2']

###############################################################################

# Plot a subset of the time-domain waveform.
fig, ax_ = plt.subplots(1, 1)
start_index = 0
stop_index = 50000
print(start_index, " ", stop_index)

t0 = t_[start_index:stop_index]
t1 = t0 - np_mean(t0)

v1 = v_[start_index:stop_index]

ax_.plot(t1, v1)
ax_.set_xlabel('Time (s)')
ax_.set_ylabel("Voltage (V)")
ax_.grid(True)
fig.show()

savefigure(fig, '10MHz_Random_Data_20Mbps', path='.')

###############################################################################

# Plot the sampled eye diagram.
ui = 1./20e6 # Unit Interval (a.k.a. bit time)
vref = 1.6 # Vref is used as the zero crossing referencing.

# AutoCenter fails on this waveform, because the eye is essentially closed.
fig = plotEye(t_*1e9, v_, ui*1e9, vref=vref, mode='edge')
ax = fig.axes[0]
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Voltage (V)')
fig.show()

savefigure(fig, '10MHz_Random_Data_Eye_20Mbps', path='.')

###############################################################################

plt.show()