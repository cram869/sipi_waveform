#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numpy import median
from sipi_waveform.wfm import wfmbinaryreader
from sipi_waveform.tracemath import plotEye

###########################################################
# Load the file.
fn = "equalized_16gbps_backplane_trace.wfm"
label = "16 Gbps Backplane Trace" # Not used.
t_, v_ = wfmbinaryreader(fn)

###########################################################
# Plot a subset of the time-domain waveform.
fig, ax_ = plt.subplots(1, 1)
start_index = -5000
stop_index = -4900
print(start_index, " ", stop_index)

t0 = t_[start_index:stop_index]*1e12
t1 = t0 - median(t0)

v1 = v_[start_index:stop_index]*1e3

ax_.plot(t1, v1)
ax_.set_xlabel('Time ($\mu$s)')
ax_.set_ylabel("Voltage (mV)")
ax_.grid(True)

fig.show()
###########################################################

###########################################################
# Plot the sampled eye diagram.
ui = 1./16e9 * 1e12 # Unit Interval (a.k.a. bit time)
vref = 0.0 # Vref is used as the zero crossing referencing.

# AutoCenter fails on this waveform, because the eye is essentially closed.
fig_eye = plotEye(t_*1e12, v_*1e3, ui, vref, AutoCenter=False)

ax_eye = fig_eye.axes[0]
ax_eye.set_xlabel('Time (ps)')
ax_eye.set_ylabel('Voltage (mV)')

fig_eye.show()
###########################################################

plt.show() # To avoid the script exiting before you can look at the plots
