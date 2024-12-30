#!/usr/bin/env python3
# Original Author: Michael Cracraft (macracra@us.ibm.com)
"""
Provide functions to analyze waveforms and work on array data.

Some functions here are presently duplicated in fft.py.

original author: Michael Cracraft (macracra@us.ibm.com)
"""

import sys
import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter  # , freqs

from .util import print_timing

############################################################################


def nearest_index(a, v):
    """
    Act similar to numpy.searchsorted, but find the nearest point
    rather than using a before or after approach.

    a - is the full vector
    v - is the value to search on.
    """
    if type(v) in (list, tuple, np.ndarray):
        inearest = []
        for v_ in v:
            inearest.append(nearest_index(a, v_))
        return inearest
    else:
        # The real calculation.
        return np.argmin(np.abs(np.array(a)-v))


def marker_fcn(a, v, y):
    """
    Use nearest_index to determine the closest indices.  Then, output
    the y values associated with these indices like a marker on an
    oscilloscope or spectrum analyzer.
    """
    il = nearest_index(a, v)
    return np.array(y)[il]


##############################################################################


@print_timing
def zerocross(t, v, vref, **kwargs):
    """
    Get the indices/time where the voltage crosses the reference
    voltage.

    **kwargs:

    slope - which direction to detect the crossings. Valid values are
    positive, negative, both, rise, fall (default: both)

    gettrace - return a trace containing a 0 for samples with no
    transition, 1 for samples with a positive slope transition, and -1
    for samples with a negative slope transition.  If the slope
    detection is set to only catch positive or negative, the other
    will not be present in the trace either. (default: False)

    mode - determines how the actual crossing time is reported.  If
    mode == 0 (default): calculates crossing by a linear interpolation
    of the 2D coordinates.  If mode == 1: returns the midpoint between
    the time samples as the crossing.  Otherwise, return the time
    sample immediately before the crossing.
    """

    ###############################################################
    # Keep this for debugging code using the older option names.
    for argkey in ['CrossingType', 'GetTrace', 'Mode']:
        if argkey in kwargs:
            print("%s has been removed as a kwarg.  Check for the updated argument name in tracemath.py.")
            return None
    ###############################################################

    ###############################################################
    # The following sets default values and replaces any that are
    # specified when the function is called.
    settings = dict(slope='both', gettrace=False, getIndices=False, mode=0)
    for key, val in kwargs.items():
        settings[key] = val
    ###############################################################

    N = len(t)
    zc = np.zeros(N)  # zero crossings
    for ii in range(0, N-1):
        if (v[ii+1] >= vref) and (v[ii] < vref):  # positive transition
            zc[ii] = 1
        if (v[ii+1] < vref) and (v[ii] >= vref):  # negative transition
            zc[ii] = -1

    if settings['gettrace']:
        return zc

    # Otherwise get the crossing times.
    # Rise and fall are the same as positive and negative, respectively.
    ct = settings['slope'].lower()
    bZCArray = {
        'both': lambda XA: XA != 0,
        'positive': lambda XA: XA > 0,
        'rise': lambda XA: XA > 0,
        'negative': lambda XA: XA < 0,
        'fall': lambda XA: XA < 0 }[ct](zc)

    if settings['getIndices']:
        return np.where(bZCArray)[0]


    # I have a couple of calculation options, but the linear interpolation is
    # the most accurate that I have built in so far.  A quadratic search is
    # probably overkill.
    ZCTime = np.array([])
    Mode = settings['mode']
    if Mode == 0:
        ZCTimeList = []
        for ii in range(0, N-1):
            if bZCArray[ii]:
                # Calculate the linear interpolated crossing time.
                lam0 = (vref-v[ii+1])/(v[ii] - v[ii+1])
                tZeroCrossing = lam0 * t[ii] + (1-lam0) * t[ii+1]
                ZCTimeList.append(tZeroCrossing)
        ZCTime = np.array(ZCTimeList)
    elif Mode == 1:
        ZCTime = np.array([0.5*(t[ii]+t[ii+1]) for ii in range(0,N-1) if bZCArray[ii]])
    else:
        ZCTime = np.array([t[ii] for ii in range(0,N-1) if bZCArray[ii]])

    # Build a crossing file.
    if 'filename' in kwargs:
        fmt = kwargs.get('format', '%.12g')
        filename = kwargs['filename']
        s = '\n'.join([fmt % z for z in ZCTime]) + '\n'
        with open(filename, 'w') as f:
            f.write(s)
    return ZCTime

def interval(t, v, vref, **kwargs):
    """
    Returns time locations and durations of intervals.

    See @zerocross for kwargs specifications.
    """
    #filename = None
    #if 'filename' in kwargs:
    #    filename = kwargs.pop('filename') # Don't want the have zerocross use my filename and build a crossing file too, or do I.

    zct = zerocross(t,v,vref,**kwargs)

    intervals = [((t[0]+t[1])/2., t[1]-t[0]) for t in zip(zct[:-1],zct[1:])]
    if 'filt' in kwargs:
        filt_fcn = kwargs['filt']
        intervals = filter(filt_fcn, intervals)

    # Build an interval file.
    if 'filename' in kwargs:
        fmt = kwargs.get('format', '%.12g')
        filename = kwargs['filename']
        s = '\n'.join([(fmt + ',' + fmt) % t for t in intervals]) + '\n'
        with open(filename, 'w') as f:
            f.write(s)
    intervals_a = np.array(intervals)
    return intervals_a[:,0], intervals_a[:,1] # time location and interval width

def meanui(t, v, vref = None):
    """meanui : This version will work well to estimate the UI
    with a clock signal, but it will overestimate the UI for a
    PRBS or other data pattern."""
    if not vref:
        vref = np.mean(v)

    tzc = zerocross(t, v, vref)
    dtzc = [tzc[ii]-tzc[ii-1] for ii in range(1, len(tzc))]
    return np.mean(dtzc)

def meanui_general(t, v, ui_target, vref = None, **kwargs):
    if not vref:
        vref = np.mean(v)

    tzc = zerocross(t, v, vref, **kwargs)
    dtzc = np.array([tzc[ii]-tzc[ii-1] for ii in range(1, len(tzc))])

    bitskip = np.round(dtzc / ui_target) - 1
    dtzc_adjustedby_ui = dtzc - ui_target*bitskip

    if kwargs.get('return_vector', False):
        return dtzc_adjustedby_ui
    return np.mean(dtzc_adjustedby_ui)

def mzcui(t, v, vref, ui):
    """
    Calculate the location of the mean zero crossing location relative to the
    the UI setting.  This is useful for eye centering.
    """

    zct = zerocross(t,v,vref)
    zctui = zct % ui
    return np.mean(zctui)

################################################################################

def eyetime(t, ui):
    """
    Realign the time vector based on the UI.
    """
    return (t % ui)

################################################################################

def eyewidth(t, v, ui, vref = None, voffset = None, bGetCenter = False):
    """
    Calculate the data eye width based on the crossings of the vref. This
    routine calculates the width of the zero crossing region, and subtracts it
    from the UI for an eye width measurment.  Returns a number
    with units the same as UI.  Take the returned value divided by UI for the
    percentage eye width.
    """

    if not vref: # Use the mean voltage for vref is none is specified.
        vref = 0.5*(np.min(v) + np.max(v))

    if voffset:
        mineye = min((eyewidth(t, v, ui, vref=(vref+voffset), bGetCenter=bGetCenter),
                      eyewidth(t, v, ui, vref=(vref-voffset), bGetCenter=bGetCenter)))
        return mineye

    zct = zerocross(t,v,vref) # Get the vref crossings.
    zctui = zct % ui
    zctui.sort() # sort ascending

    dtzcui = np.array([zctui[ii]-zctui[ii-1] for ii in range(0,len(zctui))])
    dtzcui[0] += ui # The first dt < 0 initially because zctui[-1] > zctui[0].
    # Add ui to it to make it positive < ui.

    ew = np.max(dtzcui)

    if bGetCenter:
        iimax = dtzcui.argmax()
        if iimax == 0 and zctui[-1] > zctui[0]:
            ec = (zctui[0] + zctui[-1]-ui)/2.
        else:
            ec = (zctui[iimax] + zctui[iimax-1])/2.
        return ew, ec

    return ew

################################################################################
def eyeheight(t, v, ui, vref = None):
    """
    Calculate the eye height at the center of the eye.
    """

    if not vref: # Use the mean voltage for vref is none is specified.
        vmin = np.min(v)
        vmax = np.max(v)
        vref = 0.5*(vmin + vmax)
        # Note: I write to stderr by default here because I used this function
        # for many SNAP optimizations and a write to stdout would have fouled
        # up SNAP's result collection.
        sys.stderr.write('vmin = %g, vmax = %g, vref = %g\n' % (vmin,vmax,vref))

    ew, ec = eyewidth(t, v, ui, vref, bGetCenter = True)
    #print ec

    # Use the zerocrossing calculator in reverse to find voltage crossings.
    tui = (t-(0.5*ui-ec)) % ui  # These shifts are just to center the eye and prevent
                                # and issue with detection at the edge of the ui.

    # Don't catch negative crossing that occur when tui wraps from t=ui to t=0.
    ref_crossing_voltages = zerocross(v, tui, 0.5*ui, slope='positive')


    HighVCrossings = [ v for v in ref_crossing_voltages if v > vref ]
    #print( "HighCrossings = ", HighVCrossings)
    LowVCrossings = [ v for v in ref_crossing_voltages if v < vref ]
    #print( "LowCrossings = ", LowVCrossings)

    try:
        minHigh = min(HighVCrossings)
        maxLow = max(LowVCrossings)
    except ValueError:
        sys.stderr.write('Error detecting high and/or low levels in eyeheight.\n')
        return 0.


    #print minHigh, maxLow

    eh = minHigh - maxLow
    return eh

################################################################################

def eyevectors(t, v, ui, vref = None, AutoCenter=True):
    """
    Realign the time vector and the voltage.  Make it 2 UI long.
    """
    # The mean value of the voltage should be effective if the signal is zero 
    # balanced.
    if vref is None:
        vref = np.mean(v)

    t1 = eyetime(t, ui)
    t0 = eyetime(t, ui) - ui
    tout = np.concatenate((t0,t1))
    vout = np.concatenate((v,v))

    # Note: I can't easily autocenter the voltages by shifting the voltage
    # array because the time step is not constant in the powerspice simulation.
    if AutoCenter:
        ew, ec = eyewidth(t, v, ui, vref, bGetCenter = True)
        tout -= ec

    return tout, vout

################################################################################

def eyevectors_edgealigned(t, v, ui, vref = None):
    """
    Realign the time vector and the voltage according to the leading edge 
    crossing.  This method is the same as generating a scope eye with an edge
    trigger and no delays.  Make it 2 UI long.
    """
    # The mean value of the voltage should be effective if the signal is zero 
    # balanced.
    if vref is None:
        vref = np.mean(v)

    Ts = t[1]-t[0] # sample time
    samples_per_ui_n0p25 = -int(np.round(0.5*ui/Ts))
    samples_per_ui_p0p75 = int(np.round(1.5*ui/Ts))
    i_edges = zerocross(t, v, vref, getIndices=True)

    tout = []
    vout = []
    for ii in i_edges:
        t0 = t[ii]
        start_index_test = ii + samples_per_ui_n0p25
        start_index =  start_index_test if start_index_test > 0 else 0
        end_index_test = ii + samples_per_ui_p0p75
        end_index = end_index_test if end_index_test < len(v) else -1
        tshift = t[start_index:end_index] - t0
        vshift = v[start_index:end_index]
        
        tout.extend(tshift)
        vout.extend(vshift)

    return np.array(tout), np.array(vout)

################################################################################
def plotEye(t, v, ui, vref = None, voffset = None, mode = 'ui_autocenter'):
    """
    Plot the eye for the time and voltage data given.
    """
    fig, ax = plt.subplots(1,1)

    if mode == 'ui_autocenter':
        tout, vout = eyevectors(t, v, ui, vref, voffset, AutoCenter=True)
    elif mode == 'ui':
        tout, vout = eyevectors(t, v, ui, vref, voffset, AutoCenter=False)
    elif mode.lower() == 'edge':
        tout, vout = eyevectors_edgealigned(t, v, ui, vref = None)
    else:
        print("Invalid mode argument")

    ax.plot(tout, vout, 'C0.', alpha=0.1)
    ax.grid(True)
    return fig

################################################################################

#def calcDirEyeWidth(outparam, ui, vref = None, directory='.'):
#    """
#    Calculate the eye widths for each raw file in the directory specified.
#    Output a list of tuples containing the raw file name minus extension,
#    and the eye width (percentage) and eye center point.
#    """
#    import os
#
#    rawlist = filter(lambda x: '.raw' in x, os.listdir(directory))
#
#    eyewidths = []
#    for iraw in rawlist:
#        rawname = iraw[:iraw.find('.raw')]
#
#        o1 = readraw(iraw)
#        t = o1['time']
#        v = o1[outparam]
#        ew, ec = eyewidth(t, v, ui, vref, bGetCenter = True)
#        eyewidths.append((rawname, ew, ec))
#        print (rawname, ew/ui, ec)
#
#    return eyewidths

##############################################################################
# Impedance Calculations #####################################################
##############################################################################


def v2z(v, zo, vp, vzero=0):
    """Convert a voltage waveform to an impedance.
    v => voltage array (one-dimensional numpy.ndarray)
    zo => the reference impedance
    vp => initial pulse height (might also be called v50ohm)
    vzero => zero voltage that may not be exactly zero -- subtract from the
        voltage step to get the true height of the step.
    """

    vp0 = vp - vzero
    v0 = v-vzero
    return zo * v0 / (2*vp0 - v0)


def v2z_t(t, v, tzero=None, tvp=None, zo=50):
    """Convert a voltage waveform to an impedance.  Use time windows
    to get the zero voltage and the pulse height prior to calculating
    the impedance.

    tzero => is a tuple or list containing the range of times to use
    to test for vzero.

    tvp => is the range of times to use for testing the pulse height
    to use in the conversion.
    """

    vzero = 0
    if tzero and type(tzero) in (tuple, list):
        izero = np.searchsorted(t, tzero)
        if len(izero) == 2:  # Treat like a window.
            vzero = np.mean(v[izero[0]:izero[1]])
        else:
            vzero = np.mean(v[izero])

    vp = 0.250  # this is approximately the step height on my 80E10 heads.
    if tvp and type(tvp) in (tuple, list):
        ivp = np.searchsorted(t, tvp)
        if len(ivp) == 2:  # Treat like a window.
            vp = np.mean(v[ivp[0]:ivp[1]])
        else:
            vp = np.mean(v[ivp])

    return v2z(v, zo, vp, vzero)


@np.vectorize
def v_at_zload(vp, zload, zo):
    """
    This is a scalar calculation, scalar meaning not meant for time traces, of
    the voltage level expected to be seen for a particular load impedance. The
    purpose is to figure out a voltage level for windowing to match to a
    particular non-50-ohm structure.
    """
    vo = vp*((zload-zo)/(zload+zo) + 1.)
    return vo

##############################################################################
# Rise/Fall Edge Calculations ################################################
##############################################################################


@print_timing
def slopes(t, v, vref, **kwargs):
    """
    Find the slopes of the zero crossings based on some number of
    samples around the zero crossing points.

    **kwargs:

    samplewin - is the number of samples about the crossing to
    consider in the slope calculation. (default: 10)

    twin - is the time window about the crossing to consider in the
    slope calculation. If twin is specified and not None, it will
    trump samplewin and be used for the slope calculation. (default:
    None)

    Other kwargs are defined in zerocross.  However, there are some
    options in zerocross that will cause at least a logical error in
    this routine, i.e. gettrace=True or getIndices=False.

    Outline for the procedure.
    1.) Find the zero crossings.
    2.) For each crossing ...
    2.a.) Get adjacent points determined by samplesize or edgelength
        (units of time).
    2.b.) Calculate a linear interpolation.
    2.c.) Add the slope to an array.
    """

    # The following sets default values and replaces any that are
    # specified when the function is called.
    settings = dict(slope='positive', gettrace=False, getIndices=True,
                    mode=0, samplewin=11, twin=None)
    for key, val in kwargs.items():
        settings[key] = val

    izct = zerocross(t, v, vref, **settings)  # get the zero crossings

    if settings['twin']:
        Ts = t[1]-t[0]
        ii_delta = settings['twin']/Ts
        ii_less = np.floor(ii_delta/2.)
        ii_more = np.floor(ii_delta/2.)
        i_min_v = izct-ii_less
        i_max_v = izct+ii_more
        if i_min_v[0] < 0:
            i_min_v[0] = 0
        if i_max_v[-1] > len(t)-1:
            i_max_v[-1] = -1
        slopes = np.zeros(izct.shape)
        for ii, ipair in enumerate(zip(i_min_v, i_max_v)):
            if (ipair[1]-ipair[0]) < 4:
                print("Warning: Linear fit is based on 3 or less points.")
            pcoeff = np.polyfit(t[ipair[0], ipair[1]+1],
                                v[ipair[0], ipair[1]+1], 1)
            slopes[ii] = pcoeff[0]
        return slopes
    else:
        samplewin = kwargs['samplewin']
        imaxdelta = samplewin/2
        # Bump this by one if the window is odd.
        imindelta = (samplewin/2 + samplewin % 2)
        slopes = np.array([np.polyfit(t[izct0-imindelta:izct0+imaxdelta+1],
                                      v[izct0-imindelta:izct0+imaxdelta+1],
                                      1)[0] for izct0 in izct])
        return slopes


################################################################################
################################################################################
################################################################################

def slopereversal(y, x = None, **kwargs):
    """
    Determine the location of slope reversals in a waveform.  This could be for
    peak and null detection or perhaps useful for looking at data edges if I add
    some more intelligence to the routine to exclude slope reversals of the
    signal outside of a low and high threshold.
    """

    dy = y[1:]-y[:-1]
    if x is None:
        xnew = np.arange(len(dy))
    else:
        xnew = x[:-1]

    slrvl = zerocross(xnew, dy, vref=0, **kwargs)

    return slrvl

# def peakdetect(y, x=None, **kwargs):
#     """
#     Use the @slopereversal method to track peaks in the waveform based on height
#     above the other points.
#     """
#     N = kwargs.get('N', 10) # Check adjacement points within N points
#     threshold = kwargs.get('threshold', 10.) # Must be 10 dB ?? above average to continue as a peak value.
#     kwargs['slope'] = 'fall' # Only catch peak transitions.
#     xpeak = slopereversal(y, x, **kwargs)
#     xpeak_tested = []
#     ypeak_tested = []
#     for xpi in xpeak:
#         ii_xpi = nearest_index(x, xpi)
#         ii_xpi_minus_N = ii_xpi-N
#         ii_xpi_plus_N = ii_xpi+N
#         if ii_xpi_minus_N < 0:
#             ii_xpi_plus_N = 2*N
#             ii_xpi_minus_N = 0
#         elif ii_xpi_plus_N >= len(x):
#             ii_xpi_plus_N = len(x)-1
#             ii_xpi_minus_N = len(x)-1-N
#         ymean = np.mean(y[ii_xpi_minus_N:ii_xpi_plus_N])
#         ypi = y[ii_xpi]
#         xpi_ = x[ii_xpi] # non interpolated point
#         if ypi - threshold > ymean:
#             xpeak_tested.append(xpi_)
#             ypeak_tested.append(ypi)
#     return np.array(xpeak_tested), np.array(ypeak_tested)

# def peakdetect(y, x=None, N=10, threshold=5., **kwargs):
#     """
#     Use a sliding window through the y list/array to look for spikes within the
#     window.  If the max point in the window is a user defined amount above the
#     mean by a threshold, the point is entered as a peak.
#     """
#     xpeak = []
#     ypeak = []
#     y_ = list(y) + [y[-1],]*N
#     for ii in range(len(y)):
#         ywin = y_[ii:ii+N]
#         ymean = np.mean(ywin)
#         imax = np.argmax(ywin)
#         ymax = ywin[imax]
#         if ymax > ymean + threshold:
#             ypeak.append(ymax)
#             xpeak.append(x[ii+imax])
#     return np.array(xpeak), np.array(ypeak)

@print_timing
def peakdetect(y, x, K=5, th=5.):
    """
    Use a sliding window through the y list/array to look for spikes within the
    window.  If the max point in the window is a user defined amount above the
    mean by a threshold, the point is entered as a peak.

    K : indicates the number of points less than and more than the cursor
    (present index) to include in the window.

    th : is the threshold above which a sample may be considered to be a peak.
    """
    y_ = [y[0],]*K + list(y) + [y[-1],]*K
    ymean = []
    ystd = []
    for ii in range(len(y)):
        ywin = y_[ii:ii+1+2*K]
        ymean.append(np.mean(ywin))
        ystd.append(np.std(ywin))
    ymean = np.array(ymean)
    ystd = np.array(ystd)
    ymetric = ((y - ymean) > th) * 1.
    #y2 = np.abs(y - ymean) - 3*ystd

    # Once the metric is determined, the code sweeps through and collects
    # adjacent samples classed as potential peaks.  Then, the max of each
    # set is reported as a peak.
    in_peak_set = False
    peak_sets_l = []
    for ii, yi in enumerate(ymetric):
        if yi == 1:
            if not in_peak_set:
                in_peak_set = True
                peak_sets_l.append([])
                peak_sets_l[-1].append(ii) # Start a new set.
        else:
            if in_peak_set:
                in_peak_set = False
                peak_sets_l[-1].append(ii) # Add the index past the last element
                    # and close the set.
    if in_peak_set: # If we are still in the middle of a peak set, close it.
        in_peak_set = False
        peak_sets_l[-1].append(len(ymetric))

    # Finish searching through the bundles.
    xpeaks = []
    ypeaks = []
    for i_set in peak_sets_l:
        ipeak = np.argmax(y[i_set[0]:i_set[1]])
        xpeaks.append(x[ipeak+i_set[0]])
        ypeaks.append(y[ipeak+i_set[0]])

    return np.array(ypeaks), np.array(xpeaks)

################################################################################
##

def fill_in(xout, xin, yin):
    ind_out = nearest_index(xin, xout)
    #return ind_out
    yout = [yin[ind_i] for ind_i in ind_out]
    return yout

@print_timing
def envelope_detect(yin, dx=1.0, tau_c=2., tau_d=5.0):
    A = 1 - dx/tau_d
    B = 1./tau_c

    yo = [yin[0],]
    for yi in yin[1:]:
        yprev = yo[-1]
        ytmp = A*yprev
        if yi > yprev:
            ytmp += B*(yi - yprev)
        yo.append(ytmp)
    return yo

################################################################################

def lpfilter(t,v,f_cutoff,order=4):
    Fs = 1./(t[1]-t[0]) # sample frequency
    f_nyquist = Fs/2.0
    Wn = f_cutoff / f_nyquist
    b,a = butter(order, Wn, 'low')
    v_filtered = lfilter(b,a,v-v[0]) + v[0]
    return v_filtered

################################################################################
@print_timing
def reduce_trace(x,y,step=10, yfcn=np.max, xfcn=np.mean):
    """
    Reduce the values in the ...
    Use a sliding window through the y list/array to look for ... window.

    step : indicates the number of points less than and more than the cursor
    (present index) to include in the window.

    """
    Nsamples = len(y)
    Nwins = int( Nsamples/step if Nsamples % step == 0 else Nsamples/step + 1 )

    yout = []
    xout = []
    for iwin in range(Nwins-1):
        ywin = y[iwin*step:(iwin+1)*step]
        xwin = x[iwin*step:(iwin+1)*step]
        yout.append(yfcn(ywin))
        xout.append(xfcn(xwin))
    ywin = y[(Nwins-1)*step:-1]
    xwin = x[(Nwins-1)*step:-1]
    yout.append(yfcn(ywin))
    xout.append(xfcn(xwin))


    return np.array(xout), np.array(yout)

################################################################################
################################################################################
################################################################################

# FFT Related FUNCTIONS
# My own dft calculations (very slow and meant only for debug)
def dft_test(x):
    N = len(x)
    Xf = np.zeros(N, dtype=complex)
    expfactor = 2.0j*np.pi/float(N)
    for k in range(0,N):
        Xf[k] = sum([x[m]*np.exp(expfactor*m*k) for m in range(0,N)])
    return Xf

def idft_test(Xf):
    N = len(Xf)
    x = np.zeros(N, dtype=complex)
    expfactor = -2.0j*np.pi/float(N)
    for m in range(0,N):
        x[m] = sum([Xf[k]*np.exp(expfactor*m*k) for k in range(0,N)])/float(N)
    return x

# Even/odd checks.
def isodd(m):
    if m % 2 == 0:
        return False
    return True

# Frequency translation.
# Use scipy.fftpack.fftfreq to make this simpler.
def fftfreq(N, Ts, DoubleSided = True):
    '''N = Number of points, Ts = sample rate'''
    if DoubleSided:
        return sfft.fftshift(sfft.fftfreq(N, Ts))
    else:
        Nsingle = 0
        if isodd(N):
            Nsingle = int((N+1)/2)
        else:
            Nsingle = int(N/2)
        return sfft.fftfreq(N, Ts)[:Nsingle]

def dftcalc(t, v, N=None, DoubleSided=True, **kwargs):
    """
    N = Number of samples to use in the calculation
    DoubleSided = Whether to produce a single or double sided
    response.

    Note: The non-DC terms are doubled in single-response.
    """
    if 'fs' in kwargs:
        t_,v_ = pad_for_frequency_step(t, v, kwargs['fs'])
    else:
        t_,v_ = t,v

    L = len(v_)
    if N is None:
        N = L

    Ts = t_[1]-t_[0] # Assume a constant sample rate.

    f = fftfreq(N, Ts, DoubleSided)
    if DoubleSided:
        V = sfft.fftshift(sfft.fft(v_,N))
    else: # single sided spectrum
        V = sfft.fft(v_,N)[:len(f)]*2.
        V[0] /= 2.
    return f, V/L # Divide by L to get the right scaling

@print_timing
def singlesidedspectrum(t,v, N=None, **kwargs):
    return dftcalc(t,v,N,DoubleSided=False, **kwargs)

@print_timing
def doublesidedspectrum(t,v, N=None, **kwargs):
    return dftcalc(t,v,N,DoubleSided=True, **kwargs)

@print_timing
def signal_derivative(t,v):
    Ts = t[1]-t[0]
    return np.hstack((np.array([0.,]), (v[1:]-v[:-1])/Ts))

@print_timing
def step_spectrum(t,v, **kwargs):
    t_ = t[:]
    v_diff = signal_derivative(t,v)

    f,V_diff = singlesidedspectrum(t_,v_diff, **kwargs)
    V = V_diff / (np.pi*2.*f)
    return f,V

@print_timing
def normalized_spectrum(t, v, vref, v_step_spectrum=False, vref_step_spectrum=True):
    """
    Take the time scale, @t, and two voltage waveforms, @v and @vref.  Then,
    find the single-sided spectrum of each and divide the spectrum of @v by the
    spectrum of @vref.  Use step spectrum calculations depending on type of
    response given.  @vref tend to be steps in TDT based measurements.  So, the
    default is True in this case. @v will not look like steps in case of TDR,
    cross talk, or mode conversion.  So, the default is False.  In TDT cases
    the @step_spectrum value should be set to True also.

    returns @f, @Vnorm

    If @v is a list of waveforms, @Vnorm will also be a list.
    """
    if vref_step_spectrum:
        f, Vref = step_spectrum(t, vref)
    else:
        f, Vref = singlesidedspectrum(t, vref)

    if v_step_spectrum:
        fft_func = step_spectrum
    else:
        fft_func = singlesidedspectrum

    if type(v) in (tuple,list) and type(v[0]) in (list,tuple,np.ndarray):
        Vnorm_l = []
        for v_i in v:
            ftmp, V_i = fft_func(t, v_i)
            Vnorm_l.append((V_i/Vref))
        return f, Vnorm_l
    else:
        ftmp, V = fft_func(t,v)
        return f, (V/Vref)

def pad_for_frequency_step(t,v,fs):
    Ts = t[1]-t[0] # Assumes a uniform time step.
    N = len(t)
    Nadjusted = np.ceil(1./(fs*Ts))

    if N > Nadjusted:
        print( "Warning: Requested sampling frequency is larger than existing waveform arrays provide. Returning the arrays unaltered.")
        return t,v

    t = np.array(t) # Change to numpy arrays in case they haven't already been.
    v = np.array(v)

    Ndiff = Nadjusted-N
    #print( "Ndiff: ", Ndiff)
    tadd = np.arange(1,Ndiff+1) * Ts + t[-1]
    vadd = np.ones(Ndiff) * v[-1]

    tnew = np.hstack((t,tadd))
    vnew = np.hstack((v,vadd))
    print( "After padding v.shape = ", vnew.shape)
    return tnew, vnew

def sample_for_max_frequency(t,v,Fmax):
    Ts = 1./(2*Fmax) # This would give you Nyquist rate.  Probably could oversample a bit.
    Ts_previous = t[1]-t[0]

    if Ts < Ts_previous:
        print( "Warning: Requested sampling rate is smaller than existing.  This function will not interpolate.  Returning original arrays.")
        return t,v

    k = int(np.floor(Ts/Ts_previous))

    tn = np.array(t)[::k]
    vn = np.array(v)[::k]
    return tn,vn

def zero_outside_window(t,v,twin,voutside=None):
    tmin, tmax = twin[0], twin[1]
    imin, imax = np.searchsorted(t, [tmin,tmax])
    print( "len(t):" )
    print( str(len(t)) )
    print( "zow: %d, %g, %d, %g, %d" % (imin, tmin, imax, tmax, len(t)))

    vtmp = np.array(v[:]) # make a copy to make sure I don't destroy the original.

    # searchsorted will return an index at the end of the length of the present
    # vector if the value seared is larger than the maximum value.  This will
    # cause an index exception in the present usage.  Therefore decrease imax by
    # 1 if it exceeds the last index.
    if imax >= len(t):
        imax = len(t)-1 # or -1 would work as well.

    #print imin, imax
    # if voutside is not None:
    #     vtmin = voutside
    #     vtmax = voutside
    # else:
    #     vtmin = v[imin]
    #     vtmax = v[imax]

    if voutside is None:
        vtmin = vtmp[imin]
        vtmax = vtmp[imax]
    elif type(voutside) is str and voutside.lower() == 'shift':
        vtmean = 0.5*(vtmp[imin]+vtmp[imax]) # Hopefully these are similar.
        vtmin = 0
        vtmax = 0
        vtmp -= vtmean
    else:
        vtmin = voutside
        vtmax = voutside

    vpre = vtmin*np.ones(imin)
    vpost = vtmax*np.ones(len(vtmp)-imax)
    vwin = vtmp[imin:imax]
    return np.hstack((vpre, vwin, vpost))

################################################################################
def reverse_waveform(x, y):
    """Reverse the waveform.  Equivalent to returning y' = y(-x).  Both the x
    and y arrays will be reversed."""
    x_ = np.array(x)[::-1] * (-1)
    y_ = np.array(y)[::-1]
    return x_,y_

def shift_waveform(x, y=None, x0=0.0):
    """Shift the waveform.  This routine only acts on the x array, but can
    accept a y array.  If the y array is given, it is returned unchanged with
    the exception that it is returned as a numpy.ndarray if it was not already.

    x0 is the shift applied to the function.  Effectively y' = y(x-x0).
    """
    x_ = np.array(x) - x0
    if y is None:
        return x_
    else:
        return x_, np.array(y)

##################################################
def dBmag(x):
    return 20.*np.log10(np.abs(x))
dB = dBmag

def dBpwr(x):
    return 10.*np.log10(np.abs(x))

###############################################################################
def coefficient_of_determination(y, f):
    """Coefficient of determination, or R^2, may have different definitions
    based on the source.  This function is based on a wikipedia article.
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    y: data or exact function
    f: fitted or testing function

    """
    ya = np.array(y)
    ymean = np.mean(ya)
    ss_tot = np.sum(y - ya)

    fa = np.array(f)
    ei = ya - fa
    ss_res = np.sum(ei)

    Rsquared = 1 - ss_res / ss_tot
    return Rsquared



###############################################################################
# Short Term DFT/FFTs
# TODO: Write this algorithm to take a generator or iterator to get each set of data to run
# the FFT on.
# def stfft(filename, **kwargs):
#     """
#     kwargs:
#     Nwin or (Fspan and Fcenter or Fstart and Fstop) can be set
#     Noverlap
#     """
#
#      Get the file data.
#     print( "Loading %s ..." % filename)
#     try:
#         t, v = wfm.tekcsvread(filename) # Just assume tek csv for now.
#     except:
#         print( "*** Error opening %s ***" % filename)
#     print( "... done loading %s" % filename)
#
#
#
# def stfft0(filename, Nwin, Noverlap=0, TimeLoc='mean', **kwargs):
#      Get data file handle
#     try:
#         f = open(filename, 'r')
#     except:
#         print( "Toast stfft0")
#
#      First time around grab Nwin data points. Should already be open, so no need to give the filename.
#     f, t, v = tekcsvread(points = Nwin, fileObj = f) # t,v is the list of lists for the time-voltage pairs.
#
#      Create a frequency vector, determined by the window points, Nwin.
#     if len(t) > 0:
#         Ts = t[1]-t[0]
#         print( "Ts = %g" % Ts)
#     tlist = []
#     farray = fftfreq(Nwin, Ts, DoubleSided = False)
#     print( "Fmin = %g, Fmax = %g, Fstep = %g" % (min(farray), max(farray), farray[1]-farray[0]))
#     Nsingle = len(farray)
#
#     Vf = np.zeros([0, Nsingle])
#
#     NoMoreData = False
#     while not NoMoreData:
#         try:
#             tpoint = {'start': t[0],
#                       'mean': (t[0]+t[-1])/2.,
#                       'end': t[-1]}
#             tlist.append(tpoint[TimeLoc])
#
#             if 'nodc' in kwargs:
#                 v -= np.mean(v)
#             Vf = np.vstack([Vf, sfft.fft(v)[:Nsingle]])
#
#              Get new points for the next loop.
#             if Noverlap > 0:
#                 Nadd = Nwin-Noverlap
#                 f, tnew, vnew = tekcsvread(points = Nadd, fileObj = f)
#                 if len(tnew) < Nadd:
#                     Nadd = len(tnew)
#                     NoMoreData = True
#                 t, v = np.concatenate((t[-Noverlap:],tnew[:Nadd])),\
#                        np.concatenate((v[-Noverlap:],vnew[:Nadd]))
#             else:
#                 f, t, v = tekcsvread(points = Nwin, fileObj = f)
#                 if len(t) == 0:
#                     NoMoreData = True
#         except KeyboardInterrupt:
#             print( "Caught Ctrl-C"            )
#             break
#
#     f.close()
#     print( "Time Steps Calculated = %d" % len(tlist))
#     print( "Time Step Delta = %g" % (tlist[1]-tlist[0]))
#
#      Return type is good for a contour plot with t,f,Vf as arguments.
#     return np.array(tlist), farray, Vf.transpose()

######################################################################
######################################################################
######################################################################

## # Short Term DFT/FFTs
## def stfft(tall, vall, Nwin, Noverlap=0, TimeLoc='mean'):
##     # Get data file handle
##     try:
##         f = open(filename, 'r')
##     except:
##         print( "Toast stfft0")

##     # Create a frequency vector, determined by the window points, Nwin.
##     if len(tall) > 0:
##         Ts = tall[1]-tall[0]
##         print( "Ts = %g" % Ts)
##     else:
##         print( "No time data given")
##         return

##     tlist = []
##     farray = fftfreq(Nwin, Ts, DoubleSided = False)
##     print( "Fmin = %g, Fmax = %g, Fstep = %g" % (min(farray), max(farray), farray[1]-farray[0]))
##     Nsingle = len(farray)

##     Vf = np.zeros([0, Nsingle])

##     Nleftover = len(vall) % Nwin # it is unlikely that the arrays will
##                                  # be the perfect size for the
##                                  # window. Adjust the voltage by
##                                  # padding with zeros, but just
##                                  # recreate the time vector entirely
##                                  # using the number of points and the
##                                  # sample time.  This does not allow
##                                  # for variable sample time, but I
##                                  # don't think that should be a
##                                  # problem here.  It would mess up the
##                                  # fft calculation anyway.
##     vall.extend([0,] * (Nwin-Nleftover)) # pad with zeros
##     Nsteps = len(vall) / Nwin # should be an even integer now
##     Ntotal = len(vall)
##     tall = list(np.array(0,Ntotal)*(Ts/Ntotal)) # recreate the time

##     for ii in range(0,Nsteps):
##     NoMoreData = False
##     while not NoMoreData:
##         try:
##             tpoint = {'start': t[0],
##                       'mean': (t[0]+t[-1])/2.,
##                       'end': t[-1]}
##             tlist.append(tpoint[TimeLoc])
##             Vf = np.vstack([Vf, sfft.fft(v)[:Nsingle]])

##             # Get new points for the next loop.
##             if Noverlap > 0:
##                 Nadd = Nwin-Noverlap
##                 f, tnew, vnew = tekcsvread(points = Nadd, fileObj = f)
##                 if len(tnew) < Nadd:
##                     Nadd = len(tnew)
##                     NoMoreData = True
##                 t, v = np.concatenate((t[-Noverlap:],tnew[:Nadd])),\
##                        np.concatenate((v[-Noverlap:],vnew[:Nadd]))
##             else:
##                 Nend = Nlast+Nwin
##                 Npad = 0
##                 if Nend >= len(tall):
##                     Nend
##                     Nend = len(tall)-1

##                 t = tall[Nlast:(Nlast+Nwin)]
##                 v = vall[Nlast:(Nlast+Nwin)]
##                 if len(t) == 0:
##                     NoMoreData = True
##         except KeyboardInterrupt:
##             print( "Caught Ctrl-C"            )
##             break

##     f.close()
##     print( "Time Steps Calculated = %d" % len(tlist))
##     print( "Time Step Delta = %g" % (tlist[1]-tlist[0]))

##     # Return type is good for a contour plot with t,f,Vf as arguments.
##     return np.array(tlist), farray, Vf.transpose()

######################################################################
######################################################################
######################################################################
