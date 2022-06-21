#!/usr/bin/env python
"""
Specialty tool for generating and analyzing certain types of signals.
NOT FUNCTIONAL CURRENT due to bad references to my old cramsens library.

original author: Michael Cracraft (macracra@us.ibm.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import electrical_analysis.waveform.tracemath as tm

# from matplotlib import rcParams as rcp
# rcp['text.usetex'] = True
# rcp['legend.fontsize'] = 12
# rcp['figure.figsize'] = (8,9./2.)
# rcp['savefig.dpi'] = 150

# These global settings below were concocted for use on 6 Gbps SAS links.
# 5 - 10% rise fall match about as good as can do.
tr = 60e-12
tf = 60e-12
ui = 1./(6e9)
tmax = 100*ui
fmax = 20e9
df = 10.e6
Ts = 1./(2*fmax)
bitstr = '01'*60
tskew = 0.01 * ui

def unitstep(t, t0):
    return (t >= t0) * 1.0

def unitramp(t, t0):
    return unitstep(t,t0) * (t-t0)

# SOURCES
class source_t(object):
    def __init__(self):
        pass

    def value(self, time):
        pass

class pulse_t(source_t):
    def __init__(self, vlow, vhigh, tstart, trise, tfall, period, dutycycle=0.5):
        self.vlow = vlow
        self.vhigh = vhigh
        self.tstart = tstart
        self.trise = trise
        self.tfall = tfall
        self.period = period
        self.dutycycle = dutycycle

    def value(self, time):
        t = ((time-self.tstart) * (time>self.tstart)) % self.period
        r = unitramp # just an alias

        pulse_rise = (self.vhigh-self.vlow)/self.trise * (r(t,0) - r(t,self.trise))

        DT = self.dutycycle*self.period
        pulse_fall = (self.vhigh-self.vlow)/self.tfall * (r(t,DT) - r(t,DT+self.tfall))

        return self.vlow + pulse_rise - pulse_fall

pulse0 = pulse_t(0, 1, 0, tr, tf, 2*ui)

class diffpulse_t(source_t):
    def __init__(self, vlow, vhigh, tstart, tdelay, trise, tfall, period, dutycycle=0.5):
        self.pos = pulse_t(vlow, vhigh, tstart, trise, tfall, period, dutycycle)
        self.neg = pulse_t(-vlow, -vhigh, tstart+tdelay, tfall, trise, period, dutycycle)

    def value(self, time):
        return self.pos.value(time) - self.neg.value(time)

    def commvalue(self, time):
        return self.pos.value(time) + self.neg.value(time)

dpulse0 = diffpulse_t(0, 1, ui, 0.05*ui, tr, tf, 2*ui)


class bitstream_t(source_t):
    def __init__(self, vlow, vhigh, tdelay, trise, tfall, ui, bitstring):
        '''recognized params => vlow, vhigh, tdelay, trise, tfall, ui, bitstring'''
        self.vlow = vlow
        self.vhigh = vhigh
        self.tdelay = tdelay
        self.trise = trise
        self.tfall = tfall
        self.ui = ui
        self.bitstring = bitstring
        self.bitarray = np.array([int(x) for x in bitstring])

        # I added a zero at the beginning to show no transitions in the initial time.
        # I'm not sure I like that, but I am not certain I like the alternative either.
        #self.transitions = np.concatenate([np.array([0,]), self.bitarray[1:] - self.bitarray[:-1]])
        self.transitions = np.roll(self.bitarray, 1) - self.bitarray

        self.Nbits = len(self.bitstring)
        self.tbitstart = np.arange(self.Nbits) * self.ui # create a list of bits to work from.

    def value(self, time):
        '''tdelay shifts the time before any value is calculated.'''
        t = (time - self.tdelay) * (time>self.tdelay)
        r = unitramp # just an alias

        ibit = np.searchsorted(self.tbitstart, time, side='right')-1 # which bit(s) are we in?
        if type(ibit) == np.ndarray:
            lam0 = np.array([np.sum(self.transitions[:ibit0]) for ibit0 in ibit])
        else:
            lam0 = np.sum(self.transitions[:ibit])

        vinit_value = (1-lam0)*self.vlow + lam0*self.vhigh

        t0 = self.tbitstart[ibit] # get the offset for the ramp
        vrising = (self.vhigh-self.vlow)/self.trise * (r(t,t0) - r(t,t0+self.trise))
        vfalling = (self.vhigh-self.vlow)/self.tfall * (r(t,t0) - r(t,t0+self.tfall))
        vstep_value = (self.transitions[ibit] > 0)*vrising - (self.transitions[ibit] < 0)*vfalling
        # If the transition = 0, do nothing.  Keep the vinit_value.
        return vinit_value + vstep_value

bstream0 = bitstream_t(0, 1, 0, tr, tf, ui, '01'*20)

class diffbitstream_t(source_t):
    def __init__(self, vlow, vhigh, tdelay, trise, tfall, ui, bitstring):
        '''
        Recognized params => vlow, vhigh, tdelay, trise, tfall, ui, bitstring

        By default, assume that the negative leg is delayed relative to the
        postive leg.
        '''
        self.pos = bitstream_t(vlow*0.5, vhigh*0.5, 0, trise, tfall, ui, bitstring)
        self.neg = bitstream_t(-vlow*0.5, -vhigh*0.5, tdelay, tfall, trise, ui, bitstring)

    def value(self, time):
        return self.pos.value(time) - self.neg.value(time)

    def commvalue(self, time):
        return self.pos.value(time) + self.neg.value(time)

def delaytest():
    '''Maintain tr = 60ps, tf = 60ps with a 6 Gbps bitrate. Keep a
    clock signal for simplicity.

    Sweep the tdelay value and graph the change of the signal spectrum
    at harmonics.'''

    tr = 60e-12
    tf = 60e-12
    ui = 1./6.e9
    vhigh = 1.
    vlow = 0.
    bstring = '01'*100

    fa = np.arange(3.e9, 15.01e9, 3.e9)
    flabels = ['%d GHz' % int(fi/1.e9) for fi in fa]

    t = np.linspace(0,100*ui,10000)
    tdelay_array = np.linspace(0.001, 0.2, 100)*ui

    res_array = np.zeros((len(tdelay_array), len(fa)))
    for ii in range(len(tdelay_array)):
        td = tdelay_array[ii]
        d = diffbitstream_t(vlow, vhigh, td, tr, tf, ui, bstring)
        #v = d.value(t)
        vc = d.commvalue(t)
        fc, Vc = tm.dftcalc(t,vc, DoubleSided=False)

        dBVc = tm.dBmag(Vc)

        f_ind_list = np.array([np.argmin(np.abs(fc - fai)) for fai in fa])
        res_array[ii,:] = dBVc[f_ind_list]

    print(fc[f_ind_list])
    ii = np.argmin(np.abs(tdelay_array - 0.1*ui))
    print(ii)
    print(tdelay_array[ii])
    fal = list(fa/1e9)
    ral = list(res_array[ii,:])
    print( '\n'.join(['%0.0f, %0.3g' % v for v in zip(fal,ral)]) )

    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    for ii in range(len(fa)):
        plt.plot(tdelay_array/ui * 100, res_array[:,ii], label=flabels[ii])
    a.set_xlabel('Negative Signal Delay (\% UI)')
    a.set_ylabel('dBV')
    a.set_title('6 Gbps Clock Signal Spectra Samples w.r.t Delay Skew')
    a.legend(loc='lower right')
    a.grid(True)

    fname = 'clocksignal_spectra_wrt_delayskew_6bps'
    fig.savefig(fname + '.png')
    fig.savefig(fname + '.pdf')
    fig.show()

def risefalltest():
    '''Adjust the tf in relation to the tr with a zero delay skew and
    staying at 6 Gbps.

    Sweep the tf value and graph the change of the signal spectrum
    at harmonics.'''

    tr = 60e-12
    #tf = 60e-12
    tdelay = 0
    ui = 1./8.e9
    vhigh = 0.5
    vlow = -0.5
    bstring = '01'*100

    fa = np.arange(4.e9, 20.01e9, 4.e9)
    flabels = ['%d GHz' % int(fi/1.e9) for fi in fa]

    t = np.linspace(0,100*ui,10000)
    tfall_array = (1+np.linspace(0.001, 0.2, 50))*tr

    res_array = np.zeros((len(tfall_array), len(fa)))
    full_spec_comm = []
    full_spec_diff = []
    v_l = []
    vc_l = []
    d_l = []
    for ii in range(len(tfall_array)):
        tf = tfall_array[ii]
        d = diffbitstream_t(vlow, vhigh, tdelay, tr, tf, ui, bstring)
        d_l.append(d)
        v = d.value(t)
        v_l.append(v)
        f, V = tm.singlesidedspectrum(t, v)
        dBVd = tm.dBmag(V)
        vc = d.commvalue(t)
        vc_l.append(vc)
        fc, Vc = tm.dftcalc(t,vc, DoubleSided=False)
        dBVc = tm.dBmag(Vc)

        f_ind_list = np.array([np.argmin(np.abs(fc - fai)) for fai in fa])
        res_array[ii,:] = dBVc[f_ind_list]
        full_spec_comm.append(dBVc)
        full_spec_diff.append(dBVd)

    print(fc[f_ind_list])
    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    for ii in range(len(fa)):
        plt.plot(tfall_array/tr * 100, res_array[:,ii], label=flabels[ii])
    a.set_xlabel('Fall Time (\% Rise Time)')
    a.set_ylabel('dBV')
    a.set_title('8 Gbps Clock Signal Spectra Samples w.r.t Rise-Fall Mismatch')
    a.legend(loc='lower right')
    a.grid(True)

    fname = 'clocksignal_spectra_wrt_risefallskew_8bps'
    fig.savefig(fname + '.png')
    fig.savefig(fname + '.pdf')
    #fig.show()
    DD = dict(t=t, v=v_l, vc=vc_l, fall_to_rise_ratio=tfall_array, dBVc=full_spec_comm, dBVd=full_spec_diff, f=fc, diffbitstream=d_l)

    return DD

def combinedskew():
    tr = 60e-12
    #tf = 60e-12
    #tdelay = 0
    ui = 1./6.e9
    vhigh = 1.
    vlow = 0.
    bstring = '01'*100

    tdelay_array = np.linspace(0.001, 0.1, 50)*ui
    tfall_array = (1+np.linspace(0.001, 0.2, 50))*tr

    TD, TF = np.meshgrid(tdelay_array, tfall_array)

    fa = np.arange(1.5e9, 15.01e9, 1.5e9)
    flabels = ['%0.1f GHz' % (fi/1.e9) for fi in fa]
    fnlabels = [s.replace(' ', '').replace('.','p') for s in flabels]

    t = np.linspace(0,100*ui,10000)

    res_array = np.zeros((len(fa), len(tdelay_array), len(tfall_array)))
    for ii in range(len(tdelay_array)):
        for jj in range(len(tfall_array)):
            td = tdelay_array[ii]
            tf = tfall_array[jj]

            d = diffbitstream_t(vlow, vhigh, td, tr, tf, ui, bstring)

            vc = d.commvalue(t)
            fc, Vc = tm.dftcalc(t,vc, DoubleSided=False)
            dBVc = tm.dBmag(Vc)

            f_ind_list = np.array([np.argmin(np.abs(fc - fai)) for fai in fa])
            res_array[:,ii,jj] = dBVc[f_ind_list]

    for ii in range(len(fa)):
        fig = plt.figure()
        a = fig.add_subplot(1,1,1)
        p1 = plt.pcolormesh(TD/ui*100,TF/tr*100,res_array[ii,:,:])
        a.set_xlabel('Negative Signal Delay (\% UI)')
        a.set_ylabel('Fall Time (\% Rise Time)')
        a.set_title('6 Gbps Clock Signal Spectra Samples w.r.t Skew (f=%s)' % flabels[ii])
        a.grid(True)
        plt.colorbar(p1, format='%.2g')

        fname = 'clocksignal_spectra_combined_skew_6bps_f%s' % str(fnlabels[ii]).lower()
        fig.savefig(fname + '.png')
        fig.savefig(fname + '.pdf')
        fig.show()

def delayskew_td_sweep():
    '''Maintain tr = 60ps, tf = 60ps with a 6 Gbps bitrate. Keep a
    clock signal for simplicity.

    Sweep the tdelay value and graph the change of the signal spectrum
    at harmonics.'''

    tr = 60e-12
    tf = 60e-12
    ui = 1./6.e9
    vhigh = 1.
    vlow = 0.
    bstring = '01010101'

    t = np.linspace(0,4*ui,1000)

    tdelay_array = np.array([0.1, 1., 2., 3., 5., 10.])*1.e-12
    td_labels = ['0.1 ps', '1 ps', '2 ps', '3 ps', '5 ps', '10 ps']

    res_array = np.zeros([len(tdelay_array),len(t)])
    for ii in range(len(tdelay_array)):
        td = tdelay_array[ii]
        d = diffbitstream_t(vlow, vhigh, td, tr, tf, ui, bstring)
        #v = d.value(t)
        vc = d.commvalue(t)
        res_array[ii,:] = vc

    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    for ii in range(len(tdelay_array)):
        plt.plot(t * 1e12, 1000*res_array[ii,:], label=td_labels[ii])
    a.set_xlabel('time (ps)')
    a.set_ylabel('mV')
    a.set_title('6 Gbps Common-Mode from Clock Signal w.r.t Delay Skew')

    ylim0 = a.get_ylim()
    yticks0 = np.arange(ylim0[0], ylim0[1]+0.1, 10)
    a.set_yticks(yticks0)

    a.legend(loc='best')
    a.grid(True)

    fname = 'clocksignal_commvoltage_wrt_delayskew_6bps'
    fig.savefig(fname + '.png')
    fig.savefig(fname + '.pdf')
    fig.show()

def run_single_case_w_plots(tdelay=0,tr=tr,tf=tf,ui=ui,**kwargs):
    dstream0 = diffbitstream_t(0, 1, tdelay, tr, tf, ui, '01'*100)
    t = np.linspace(0,100*ui,11000)
    v = dstream0.value(t)
    vp = dstream0.pos.value(t)
    vn = dstream0.neg.value(t)
    vc = dstream0.commvalue(t)

    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    a.plot(t,v, label='diff')
    a.plot(t,vc, label='comm')
    a.plot(t,vp, label='pos')
    a.plot(t,vn, label='neg')

    if 'txlim' in kwargs:
        a.set_xlim(kwargs['txlim'])

    a.grid(True)
    a.set_ylabel('Voltage (V)')
    a.set_xlabel('Time (s)')
    if 'ttitle' in kwargs:
        a.set_title(kwargs['ttitle'])
    a.legend(loc='best')
    fig.show()

    if 'tfile' in kwargs:
        basename = kwargs['tfile']
        fig.savefig(basename + '.png')
        fig.savefig(basename + '.pdf')
        fig.savefig(basename + '.eps')

    f,V = tm.dftcalc(t,v, DoubleSided=False)
    fp,Vp = tm.dftcalc(t,vp, DoubleSided=False)
    fn,Vn = tm.dftcalc(t,vn, DoubleSided=False)
    fc,Vc = tm.dftcalc(t,vc, DoubleSided=False)

    dB = tm.dBmag

    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    print(f.shape, V.shape)
    a.plot(f, dB(V), label='diff')
    a.plot(fc, dB(Vc), label='comm')
    a.grid(True)
    a.set_ylabel('Voltage (dBV)')
    a.set_xlabel('Frequency (Hz)')
    if 'ftitle' in kwargs:
        a.set_title(kwargs['ftitle'])
    a.legend(loc='best')
    a.set_xlim(0,20e9)
    fig.show()

    if 'ffile' in kwargs:
        basename = kwargs['ffile']
        fig.savefig(basename + '.png')
        fig.savefig(basename + '.pdf')

    fa = np.arange(3.e9, 15.01e9, 3.e9)
    f_ind_list = np.array([np.argmin(np.abs(fc - fai)) for fai in fa])

    fcl = list(fc[f_ind_list]/1e9) # GHz
    Vcl = list(20.*np.log10(np.abs(Vc[f_ind_list]))) # dBV/m
    print( '\n'.join(['%0.0f, %0.3g' % v for v in zip(fcl,Vcl)]) )

    print( "df = %g" % (f[1]-f[0]) )

    return t, f

def make_skew_example_plots():
    """
    Need to create illustrations of the difference between skew and
    rise/fall mismatch.
    """
    # In-Pair Skew
    ui = 1./6e9
    tr = 60e-12
    tf = 60e-12
    tskew = 0.1*ui

    titlestr = 'In-Pair Skew Effects with 10\% Skew'
    filename = 'delayskew_effect'
    t,f = run_single_case_w_plots(tdelay=tskew, tr=tr, tf=tf, ui=ui, ttitle=titlestr, tfile=filename, txlim=(0.,4.5*ui))

    # Rise/Fall Mismatch
    ui = 1./6e9
    tr = 60e-12
    tf = 1.1*tr
    tskew = 0.

    titlestr = 'Rise-Fall Mismatch Effects with 10\% Mismatch'
    filename = 'risefallmismatch_effect'
    t,f = run_single_case_w_plots(tdelay=tskew, tr=tr, tf=tf, ui=ui, ttitle=titlestr, tfile=filename, txlim=(0.,4.5*ui))

def make_spectra_example_plot(random=False):
    """
    Create spectra data for in-pair skew and rise-fall mismatch and plot the
    combined effects as well.
    """

    tr = 60e-12
    tf = 60e-12
    ui = 1./6.e9
    vhigh = 1.
    vlow = 0.
    if not random:
        bp = '01'*1000
    else:
        bp = [{True:'1',False:'0'}[b] for b in (np.random.random(1000)>0.5)]

    dstream_skew = diffbitstream_t(vlow, vhigh, 0.1*ui, tr, tf, ui, bp)
    dstream_risefall = diffbitstream_t(vlow, vhigh, 0, tr, tr*1.1, ui, bp)
    dstream_both = diffbitstream_t(vlow, vhigh, 0.1*ui, tr, tr*1.1, ui, bp)

    t = np.linspace(0, 400*ui, 40001)
    print( "Ts = ", t[1]-t[0], " : Tmax = ", max(t) )
    vc_skew = dstream_skew.commvalue(t)
    vc_risefall = dstream_risefall.commvalue(t)
    vc_both = dstream_both.commvalue(t)

    fskew,Vc_skew = tm.dftcalc(t, vc_skew, DoubleSided=False)
    frisefall,Vc_risefall = tm.dftcalc(t, vc_risefall, DoubleSided=False)
    fboth,Vc_both = tm.dftcalc(t, vc_both, DoubleSided=False)

    print( "Fs = ", fboth[1]-fboth[0], " : Fmax = ", max(fboth) )

    dB = tm.dBmag

    fig = plt.figure()
    a = fig.add_subplot(1,1,1)
    fharm = np.arange(3.e9, 18.01e9, 3.e9)

    a.plot(fboth/1e9, dB(Vc_both), label='Both', color = 'r')
    iii = np.array([np.argmin(np.abs(fboth - fharm_i)) for fharm_i in fharm])
    lboth, = a.plot(fboth[iii]/1e9, dB(Vc_both[iii]), linestyle='none', marker='s', markersize=9., label='Both', color = 'r', alpha=0.7)

    a.plot(frisefall/1e9, dB(Vc_risefall), label='10\% rise-fall mismatch', color = 'g')
    iii = np.array([np.argmin(np.abs(frisefall - fharm_i)) for fharm_i in fharm])
    lrf, = a.plot(frisefall[iii]/1e9, dB(Vc_risefall[iii]), linestyle='none', marker='d', label='10\% rise-fall mismatch', color = 'g', alpha=0.7)

    a.plot(fskew/1e9, dB(Vc_skew), label='10\% skew', color = 'b')
    iii = np.array([np.argmin(np.abs(fskew - fharm_i)) for fharm_i in fharm])
    lsk, = a.plot(fskew[iii]/1e9, dB(Vc_skew[iii]), linestyle='none', marker='o', label='10\% skew', color = 'b', alpha=0.7)

    a.grid(True)
    a.set_ylabel('Voltage (dBV)')
    a.set_xlabel('Frequency (GHz)')
    a.set_xlim((0,20))
    a.set_xticks(np.arange(0, 20, 3))
    if random:
        a.set_ylim((-100, -30))
    else:
        a.set_ylim((-80,-10))
    a.set_title('Spectral Effects of Non-Ideal Differential Signals')
    a.legend((lsk, lrf, lboth), ('10\% skew','10\% rise-fall mismatch','Both'), loc='best')

    if not random:
        fn = 'nonideal_spectral_effects'
    else:
        fn = 'nonideal_spectral_effects_randombp'
    fig.savefig(fn + '.png')
    fig.savefig(fn + '.pdf')
    fig.savefig(fn + '.eps')

    fig.show()

    for name, f,Vc in (('skew', fskew, Vc_skew), ('risefall', frisefall, Vc_risefall), ('both', fboth, Vc_both)):
        print(name, ':')
        iii = np.array([np.argmin(np.abs(f - fharmi)) for fharmi in fharm])
        fl = list(f[iii]/1e9) # GHz
        Vcl = list(20.*np.log10(np.abs(Vc[iii]))) # dBV/m
        print( '\n'.join(['%0.3f, %0.3g' % v for v in zip(fl,Vcl)]) )
        print( "df = %g" % (f[1]-f[0]) )

    return fig
