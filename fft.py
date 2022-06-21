#!/usr/bin/env python
'''
Provide convenience functions for Fourier analysis of signals.

original author: Michael Cracraft (macracra@us.ibm.com)
'''

import os, os.path
#import pylab as pl
import scipy.fftpack as sfft
import numpy as np
import matplotlib.pyplot as plt
from electrical_analysis.waveform.tracemath import nearest_index

# def frequency_scale(f, unit='GHz'):
#     """
#     This routine assumes that the frequency argument is in Hz to start.
#     """
#     f_scale = dict(hz=1., khz=1e3, mhz=1e6, ghz=1e9)[unit.lower()]
#     return f/f_scale

# def time_scale(t, unit='ns'):
#     """
#     This routine assumes that the frequency argument is in sec to start.
#     """
#     t_scale = dict(s=1., sec=1., ms=1e3, us=1e6, ns=1e9, ps=1e12, fs=1e15)[unit.lower()]
#     return t*t_scale

# def voltage_scale(v, unit='mV'):
#     """
#     This routine assumes that the voltage argument is in V to start.
#     """
#     v_scale = dict(kv=1e-3, v=1., mv=1e3, uv=1e6)[unit.lower()]
#     return v*v_scale

def tekcsvread(filename = None, points = None, fileObj = None):
    """
    Read the csv data from the Tek Scope. Get col 4 and 5 only.
    This does not close the file on return but returns the file
    object to be reused later to continue loading for the previous
    location.
    """
    f = None
    if not fileObj or fileObj.closed:
        """ Open the file at the start. """
        try:
            f = open(filename, 'r')
        except:
            print("toast")
    else:
        """ Work with an already open file. """
        f = fileObj

    # If points was given read up to that many lines.  If points was not given,
    # read to the end of the file.
    if points > 0:
        fpointLines = [f.readline().strip().split(',')[3:] for ii in range(0,points)]
        tlist = [float(v[0]) for v in fpointLines if v]
        vlist = [float(v[1]) for v in fpointLines if v] # cull out the empty list if they exist
    else:
        fpointOneLine = f.readline().strip().split(',')[3:]
        tlist = []
        vlist = []
        while fpointOneLine:
            tlist.append(float(fpointOneLine[0]))
            vlist.append(float(fpointOneLine[1]))
            fpointOneLine = f.readline().strip().split(',')[3:]
    # Check the first record to see if it is empty.
    if len(tlist) == 0:
        return f, [], []
    # Otherwise return the file object and the point lists.
    if points is not None and len(tlist) != points:
        Nadded = points - len(tlist)
        Tmaxprev = tlist[-1]
        Ts = Tmaxprev - tlist[-2]
        Tmax = Tmaxprev + (Ts * Nadded)
        tlist += list(np.arange(Tmaxprev+Ts, Tmax+Ts/2., Ts))
        print("points ==> ", len(tlist), points)
        vlist += [0]*Nadded
    return f, np.array(tlist), np.array(vlist)

def tekcsvreadsubset(ifilename, ofilename, xrange, N = None):
    try:
        f = open(ifilename, 'r')
        fo = open(ofilename, 'w')

        if N:
            flineslist = f.read().splitlines()
            for iline in flineslist:
                xi = float(iline.strip().split(',')[3])
                if (xi >= xrange[0]) and (xi <= xrange[1]):
                    fo.write(iline + '\n')
        else:
            fpointLines = [f.readline().strip().split(',')[3:] for ii in range(0,N)]
            while len(fpointLines) > 0:
                for iline in fpointLines:
                    xi = float(iline.strip().split(',')[3])
                    if (xi >= xrange[0]) and (xi <= xrange[1]):
                        fo.write(iline + '\n')
                fpointLines = [f.readline().strip().split(',')[3:] for ii in range(0,N)]

        f.close()
        fo.close()
    except:
        print( "What's wrong with tekcsvreadsubset?" )

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
            print(Nsingle)
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
        t_,v_ = t[:],v[:]

    L = len(v_)
    if not N:
        N = L

    Ts = t_[1]-t_[0] # Assume a constant sample rate.

    f = fftfreq(N, Ts, DoubleSided)
    if DoubleSided:
        V = sfft.fftshift(sfft.fft(v_,N))
    else: # single sided spectrum
        V = sfft.fft(v_,N)[:len(f)]*2.
        V[0] /= 2.
    return f, V/L # Divide by L to get the right scaling

def singlesidedspectrum(t,v, N=None, **kwargs):
    return dftcalc(t,v,N,DoubleSided=False, **kwargs)

def doublesidedspectrum(t,v, N=None, **kwargs):
    return dftcalc(t,v,N,DoubleSided=True, **kwargs)

def signal_derivative(t,v):
    Ts = t[1]-t[0]
    return np.hstack((np.array([0.,]), (v[1:]-v[:-1])/Ts))

def step_spectrum(t,v, **kwargs):
    t_ = t[:]
    v_diff = signal_derivative(t,v)

    f,V_diff = singlesidedspectrum(t_,v_diff, **kwargs)
    V = V_diff / (np.pi*2.*f)
    return f,V

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
        print( "Warning: Requested sampling frequency is larger than existing waveform arrays provide. Returning the arrays unaltered." )
        return t,v

    t = np.array(t) # Change to numpy arrays in case they haven't already been.
    v = np.array(v)

    Ndiff = Nadjusted-N
    #print( "Ndiff: ", Ndiff )
    tadd = np.arange(1,Ndiff+1) * Ts + t[-1]
    vadd = np.ones(Ndiff) * v[-1]

    tnew = np.hstack((t,tadd))
    vnew = np.hstack((v,vadd))
    print( "After padding v.shape = ", vnew.shape )
    return tnew, vnew

def sample_for_max_frequency(t,v,Fmax):
    Ts = 1./(2*Fmax) # This would give you Nyquist rate.  Probably could oversample a bit.
    Ts_previous = t[1]-t[0]

    if Ts < Ts_previous:
        print( "Warning: Requested sampling rate is smaller than existing.  This function will not interpolate.  Returning original arrays." )
        return t,v

    k = int(np.floor(Ts/Ts_previous))

    tn = np.array(t)[::k]
    vn = np.array(v)[::k]
    return tn,vn

def zero_outside_window(t,v,twin,voutside=None):
    tmin, tmax = twin[0], twin[1]
    imin, imax = np.searchsorted(t, [tmin,tmax])
    print( "len(t):" )
    print( len(t) )
    print( "zow: %d, %g, %d, %g, %d" % (imin, tmin, imax, tmax, len(t)) )

    vtmp = np.array(v[:]) # make a copy to make sure I don't destroy the original.

    # searchsorted will return an index at the end of the length of the present
    # vector if the value seared is larger than the maximum value.  This will
    # cause an index exception in the present usage.  Therefore decrease imax by
    # 1 if it exceeds the last index.
    if imax >= len(t):
        imax = len(t)-1 # or -1 would work as well.

    #print( imin, imax )
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
    
def step_response(f, Y):
    """Yowh"""
    Fs = f[1]-f[0]
    
    y = sfft.ifft(Y)
    return y



################################################################################

# def peakDetect(f, dBV, aboveMean):
#     """
#     Find the mean of dBV.  Look for points > mean(dBV) + aboveMean.
#     Return a frequency and a dB voltage vector of the points matching
#     these criterion.
#     """
#     meandBV = np.mean(dBV)
#
#     fpk = []
#     dBVpk = []
#     dBVthreshold = meandBV + aboveMean
#     for ii in range(0, len(f)):
#         if dBV[ii] > dBVthreshold:
#             # Look for a slope change to detect the peak.
#             if ii > 0 and ii < len(f)-1:
#                 delta1 = dBV[ii]-dBV[ii-1]
#                 delta2 = dBV[ii+1]-dBV[ii]
#                 if delta1 > 0 and delta2 < 0:
#                     fpk.append(f[ii])
#                     dBVpk.append(dBV[ii])
#             else:
#                 fpk.append(f[ii])
#                 dBVpk.append(dBV[ii])
#
#     return np.array(fpk), np.array(dBVpk)


################################################################################
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
#     print "Loading %s ..." % filename
#     try:
#         t, v = wfm.tekcsvread(filename) # Just assume tek csv for now.
#     except:
#         print "*** Error opening %s ***" % filename
#     print "... done loading %s" % filename
#
#
#
# def stfft0(filename, Nwin, Noverlap=0, TimeLoc='mean', **kwargs):
#      Get data file handle
#     try:
#         f = open(filename, 'r')
#     except:
#         print "Toast stfft0"
#
#      First time around grab Nwin data points. Should already be open, so no need to give the filename.
#     f, t, v = tekcsvread(points = Nwin, fileObj = f) # t,v is the list of lists for the time-voltage pairs.
#
#      Create a frequency vector, determined by the window points, Nwin.
#     if len(t) > 0:
#         Ts = t[1]-t[0]
#         print "Ts = %g" % Ts
#     tlist = []
#     farray = fftfreq(Nwin, Ts, DoubleSided = False)
#     print "Fmin = %g, Fmax = %g, Fstep = %g" % (min(farray), max(farray), farray[1]-farray[0])
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
#             print "Caught Ctrl-C"
#             break
#
#     f.close()
#     print "Time Steps Calculated = %d" % len(tlist)
#     print "Time Step Delta = %g" % (tlist[1]-tlist[0])
#
#      Return type is good for a contour plot with t,f,Vf as arguments.
#     return np.array(tlist), farray, Vf.transpose()

######################################################################
######################################################################
######################################################################

# Short Term DFT/FFTs
def stfft(t, f, twin, toverlap=0., TimeLoc='mean', fmax=None, RemoveDC=False):
    # Create a frequency vector, determined by the window points, Nwin.
    Ts = t[1]-t[0] # Assumes a constant sample rate.
    print( "Ts = %g" % Ts )

    Nwin = int(np.ceil(twin/Ts))
    Noverlap = int(np.floor(toverlap/Ts))
    Nwin_minus_overlap = Nwin - Noverlap

    farray_tmp = fftfreq(Nwin, Ts, DoubleSided = False)
    if fmax:
        imax = nearest_index(farray_tmp, fmax)
        farray = farray_tmp[:imax]
    else:
        farray = farray_tmp[:]

    if RemoveDC:
        farray = farray[1:]

    print( "Fmin = %g, Fmax = %g, Fstep = %g" % (min(farray), max(farray), farray[1]-farray[0]) )
    Nsingle = len(farray)

    if RemoveDC:
        F = np.zeros([0, Nsingle-1])
    else:
        F = np.zeros([0, Nsingle])

    # total length should be (Nwin-Noverlap)*Nsteps + Noverlap
    Nleftover = (len(f) - Noverlap) % (Nwin - Noverlap)
    f = np.hstack( (f, ([0,] * (Nwin-Nleftover)) ) ) # pad with zeros
    Nsteps = int((len(f)-Noverlap) / (Nwin-Noverlap)) # should be an even integer now
    Ntotal = len(f)
    t_extended = list( np.arange(0,Ntotal)*Ts ) # recreate the time

    i0 = 0
    i1 = Nwin
    tlist = []
    for ii in range(0,Nsteps):
        try:
            t_subset = t_extended[i0:i1]
            f_subset = f[i0:i1]

            tpoint = {'start': t_subset[0],
                      'mean': (t_subset[0]+t_subset[-1])/2.,
                      'end': t_subset[-1]}
            tlist.append(tpoint[TimeLoc])

            F_subset = sfft.fft(f_subset)[:Nsingle]*2.0 # These two lines are pulled from dftcalc
            F_subset[0] /= 2.0

            if RemoveDC: F_subset = F_subset[1:]

            F = np.vstack([F, F_subset])
            # or
            # f_tmp, F_subset = dftcalc(t_subset, f_subset, DoubleSided=False)
            #F = np.vstack([F, F_subset])
            # or
            #F = np.vstack([F, sfft.fft(f_subset)[:Nsingle]]) # Not properly scaled for single-sided.

            # Get new points for the next loop.
            i0 = i1 - Noverlap
            i1 = Nwin + i0

        except KeyboardInterrupt:
            print( "Caught Ctrl-C" )
            break

    print( "Time Steps Calculated = %d" % len(tlist) )
    print( "Time Step Delta = %g" % (tlist[1]-tlist[0]) )

    return np.array(tlist), farray, F.transpose()

######################################################################
######################################################################
######################################################################

# Plotting functions for the stfft rig.
def stfftcontf(t,f,Xf, **kwargs):
    """
    kwargs:
    titstr == Title for plot.
    bandwidth == tuple for frequency band of interest
    levels == number of contour levels to plot
    xlabel == label of the x axis
    ylabel == label of the y axis
    colormap == the command from pylab to set the colormap
    colorbar == if False, don't plot the colorbar on the picture
    clim == set the color limits with a tuple
    xlim == set the time (x-axis) limits
    show == run the show command if present and true
    """

    #plt.figure()
    if 'levels' in kwargs:
        plt.contourf(t,f,Xf, kwargs['levels'])
    else:
        plt.contourf(t,f,Xf)

    if 'bandwidth' in kwargs:
        plt.ylim(kwargs['bandwidth'])

    xlab1 = 'Time (seconds)'
    if 'xlabel' in kwargs:
        xlab1 = kwargs['xlabel']
    ylab1 = 'Frequency (Hz)'
    if 'ylabel' in kwargs:
        ylab1 = kwargs['ylabel']

    plt.xlabel(xlab1)
    plt.ylabel(ylab1)

    if 'colormap' in kwargs:
        kwargs['colormap']()
    else:
        plt.jet()

    if 'colorbar' not in kwargs or kwargs['colorbar'] == True:
        plt.colorbar()

    if 'clim' in kwargs:
        plt.clim(kwargs['clim'])

    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])

    if 'titstr' in kwargs:
        plt.title(kwargs['titstr'])

    if 'show' in kwargs and kwargs['show']:
        plt.show()

# TODO: Correct for the newer version of matplotlib
## import matplotlib.axes3d as p3
## def stfftcont3d(t,f,Xf):
##     """
##     The contourf3D routine appears to be unimplemented at the moment for 2.4.
##     in mplt3D change:
##     levels, colls = self.contourf(X, Y, Z, 20)
##     to:
##     C = self.contourf(X, Y, Z, *args, **kwargs)
##     levels, colls = (C.levels, C.collections)
##     """

##     fig=np.figure()
##     ax = p3.Axes3D(fig)
##     ax.contourf3D(t,f,Xf)
##     ax.set_xlabel('Time (seconds)')
##     ax.set_ylabel('Frequency (Hz)')
##     #    ax.set_zlabel('')
##     fig.add_axes(ax)
##     np.show()

##################################################



##################################################
def dBmag(x):
    return 20.*np.log10(np.abs(x))
dB = dBmag

def dBpwr(x):
    return 10.*np.log10(np.abs(x))

################################################################################
def WriteGnuplotFile(filename, x, y, Z):
    """

    """
    from copy import deepcopy

    if len(np.shape(x)) > 1:
        X = deepcopy(x)
        x = X[0]

    if len(np.shape(y)) > 1:
        Y = deepcopy(y)
        y = np.transpose(Y)[0]

    Nx = len(x)
    Ny = len(y)

    xlist = sorted(list(x) * Ny)
    ylist = list(y) * Nx
    zlist = []
    for icol in np.transpose(Z):
        zlist.append(list(icol))

    return xlist, ylist, zlist, Nx, Ny
