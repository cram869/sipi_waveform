#!/usr/bin/env python
"""
Specialty tool for calculating progressive FFTs of a function.
NOT FUNCTIONAL CURRENT due to bad references to my old cramsens library.

original author: Michael Cracraft (macracra@us.ibm.com)
"""

import cramsens.mcffttools as F
import cramsens.wfm as W
import cramsens.touchstone as ts
import cramsens.plot_tools as P
plt = P.plt
import cramsens.units as U

import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

#from tdr_analysis import *

def set_class_attributes(obj, **kwargs):
    """
    Reassign any attributes in obj that are present in kwargs.  If the key is
    not present in kwargs, reassign with the original value from obj.  Also,
    pop any found values from kwargs so they are not reused later, and
    return the remain kwargs dictionary.
    """
    for k,v in obj.__dict__.iteritems():
        obj.__setattr__(k, kwargs.pop(k, v))
    return kwargs

def savefigure(fig, filename):
    pathfilename = '../plots/' + filename
    fig.savefig(pathfilename + '.png')
    fig.savefig(pathfilename + '.pdf')

dB = lambda y : 20.*np.log10(np.abs(y))

def signal_derivative(t,v):
    Ts = t[1]-t[0]
    return np.hstack((np.array([0.,]), (v[1:]-v[:-1])/Ts))

def step_spectrum(t,v):
    v_diff = signal_derivative(t,v)
    f,V_diff = F.singlesidedspectrum(t,v_diff)
    V = V_diff / (np.pi*2.*f)
    return f,V

def sliding_fft(t,v,vref,tzero,twin,WindowSteps=50,**kwargs):
    """
    kwargs : two_figures, figsize, subplot_adjust, voutside,
    step_fft_signal, step_fft_reference, funit, vunit
    """

    # Make the figures.
    fig_l = []
    if kwargs.get('two_figures', False):
        fig1 = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        fig2 = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        ax1 = fig1.add_subplot(1,1,1)
        ax2 = fig2.add_subplot(1,1,1)
        fig1.suptitle(kwargs.get('title',''))
        fig2.suptitle(kwargs.get('title',''))
        fig_l = [fig1,fig2]
    else:
        fig = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        fig.suptitle(kwargs.get('title',''))
        fig_l = [fig,]
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xlabel(kwargs.get('xlabel1', ''))
    ax2.set_xlabel(kwargs.get('xlabel2', ''))
    ax1.set_ylabel(kwargs.get('ylabel1', ''))
    ax2.set_ylabel(kwargs.get('ylabel2', ''))


    # Calculate the spectrum of the reference signal.
    if kwargs.get('step_fft_reference',True):
        fref, Vref = step_spectrum(t,vref)
    else:
        fref, Vref = F.singlesidedspectrum(t,vref)

    # Plot the time signal.
    ax1.plot(U.time_scale(t, from_units='sec', to_units='ns'),
             U.voltage_scale(v, from_units='V', to_units='mv'),
             color='k')
    rect_ymin, rect_ymax = ax1.get_ylim()

    # Window points
    tsteps = np.linspace(twin[0],twin[1],WindowSteps+1)

    # Create a color mapping.
    cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=WindowSteps-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

    # Use the first window point as a reference for the outside .
    voutside = v[np.searchsorted(t,twin[0])]

    # Plot the different windowed spectrum.
    Vw_l = []
    for ii in range(len(tsteps)-1):
        vw = F.zero_outside_window(t,v,twin=(tsteps[0],tsteps[ii+1])) # TODO: integrate voutside option also.
        f, Vw_i = F.singlesidedspectrum(t, vw)
        Vw_l.append(Vw_i)

        col_i = scalarMap.to_rgba(ii)

        r1 = Rectangle((tsteps[ii]*1e9,rect_ymin),(tsteps[ii+1]-tsteps[ii])*1e9,
                       rect_ymax-rect_ymin, color=col_i, alpha=0.5)
        ax1.add_patch(r1)
        ax2.plot(U.frequency_scale(f,to_units='ghz'), dB(Vw_i/Vref), color=col_i, alpha=0.3)

    if 'xlim1' in kwargs: P.set_xlim(ax1, kwargs['xlim1'])
    if 'xlim2' in kwargs: P.set_xlim(ax2, kwargs['xlim2'])
    if 'ylim1' in kwargs: P.set_ylim(ax1, kwargs['ylim1'])
    if 'ylim2' in kwargs: P.set_ylim(ax2, kwargs['ylim2'])

    P.show_figures(fig_l)

    return [ax1,ax2],fig_l


################################################################################

def interactive_windowed_fft(t,v,vref=None,**kwargs):
    """
    kwargs : two_figures, figsize, subplot_adjust, voutside,
    step_fft_signal, step_fft_reference, funit, vunit
    """

    # Make the figures.
    fig_l = []
    if kwargs.get('two_figures', False):
        fig1 = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        fig2 = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        ax1 = fig1.add_subplot(1,1,1)
        ax2 = fig2.add_subplot(1,1,1)
        fig1.suptitle(kwargs.get('title',''))
        fig2.suptitle(kwargs.get('title',''))
        fig_l = [fig1,fig2]
    else:
        fig = plt.figure(figsize=kwargs.get('figsize',(8,6)))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        fig.suptitle(kwargs.get('title',''))
        fig_l = [fig,]
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xlabel(kwargs.get('xlabel1', ''))
    ax2.set_xlabel(kwargs.get('xlabel2', ''))
    ax1.set_ylabel(kwargs.get('ylabel1', ''))
    ax2.set_ylabel(kwargs.get('ylabel2', ''))

    # Scale the time vector.
    ts = U.time_scale(t,to_units='ns')
    vs = U.voltage_scale(v,'mv')

    # Plot the time signal.
    lmain, = ax1.plot(ts,vs,color='b',alpha=0.5,label='Raw')
    lwin, = ax1.plot(ts,vs,color='r',label='Windowed')

    twfm_l = kwargs.get('twfm', [])
    #ax0 = ax1.twinx()
    for lab_i, v_i in twfm_l:
        ltmp, = ax1.plot(ts,U.voltage_scale(v_i,'mv'),label=lab_i)

    # Calculate the spectrum of the reference signal.
    if vref is None:
        Vref = 1.
    else:
        lref, = ax1.plot(ts,U.voltage_scale(vref,'mv'),color='g')
        if kwargs.get('step_fft_reference',True):
            fref, Vref = step_spectrum(t,vref)
        else:
            fref, Vref = F.singlesidedspectrum(t,vref)


    #rect_ymin, rect_ymax = ax1.get_ylim()

    # Plot the different windowed spectrum.
    vw = F.zero_outside_window(t,v,twin=(t[0],t[-1])) # TODO: integrate voutside option also.
    if kwargs.get('step_fft', False):
        f, Vw = step_spectrum(t, vw)
    else:
        f, Vw = F.singlesidedspectrum(t, vw)

    lspec, = ax2.plot(U.frequency_scale(f,to_units='ghz'), dB(Vw/Vref), color='r', label='Windowed Spectrum')
    if kwargs.get('show_reference_spectrum', True):
        lspec2, = ax2.plot(U.frequency_scale(f,to_units='ghz'), dB(Vw), color='r', linestyle='--', label='Windowed Spectrum')

    if vref is not None and kwargs.get('show_reference_spectrum', True):
        lrefspec, = ax2.plot(U.frequency_scale(fref,to_units='ghz'), dB(Vref), color='g', label='Reference')

    if 'xlim1' in kwargs: P.set_xlim(ax1, kwargs['xlim1'])
    if 'xlim2' in kwargs: P.set_xlim(ax2, kwargs['xlim2'])
    if 'ylim1' in kwargs: P.set_ylim(ax1, kwargs['ylim1'])
    if 'ylim2' in kwargs: P.set_ylim(ax2, kwargs['ylim2'])

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    def onspanselect1(t0,t1):
        if fig_l[0].canvas.toolbar._active in ('PAN', 'ZOOM'):
            print( "Disabled for pan or zoom." )
        else:
            print( "New Window => (%g,%g), width = %g" % (t0,t1,t1-t0) )
            vw = F.zero_outside_window(t,v,twin=(t0*1e-9,t1*1e-9))
            lwin.set_ydata(U.voltage_scale(vw,'mv'))

            # Add a rectangle for the selection to ax1.
            if len(ax1.patches) > 0:
                ax1.patches.pop(0)
            ymin,ymax = ax1.get_ylim()
            r1 = Rectangle((t0,ymin),t1-t0,ymax-ymin,color='b',alpha=0.3)
            ax1.add_patch(r1)

            if kwargs.get('step_fft', False):
                f, Vw = step_spectrum(t, vw)
            else:
                f, Vw = F.singlesidedspectrum(t, vw)
            lspec.set_ydata(dB(Vw/Vref))

            for fig_i in fig_l:
                fig_i.canvas.draw()

    span1 = SpanSelector(ax1, onspanselect1,
                         'horizontal', useblit=True,
                         rectprops=dict(alpha=0.5, facecolor='blue'))

    for fig_i in fig_l:
        fig_i.canvas.draw()

    P.show_figures(fig_l)

    return [ax1,ax2],fig_l

################################################################################
# This would sort of be like the more generic form of the sliding/extending
# window.
def listed_window_fft(t,v,vref,tzero,twin_l,**kwargs):
    pass

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class VoltageWaveformAndSpectrum(object):
    """Class to represent a voltage waveform with a ..."""
    def __init__(self, t, v, **kwargs):
        self.f = None
        self.V = None
        self.use_step_spectrum = False
        self.voutside = None
        self.from_tunits = 's'
        self.tunits = 'ns'
        self.from_vunits = 'v'
        self.vunits = 'mv'
        self.funits = 'GHz'
        self.magunits = 'dB' # I'm not sure what else to do for this at the moment.
        # It would make sense to have options like dBmV.

        self.name = None

        set_class_attributes(self, **kwargs)

        self.t = U.time_scale(t, self.from_tunits, self.tunits)
        self.v = U.voltage_scale(v, self.from_vunits, self.vunits)

        # Not sure if I want to leave this here or abstract out the voltage and
        # voltage/spectrum + aux waveform classes.
        if 'v_aux' in kwargs:
            self.v_aux = U.voltage_scale(kwargs['v_aux'], self.from_vunits,
                                         self.vunits)
            self.aux_name = kwargs.get('aux_name', 'Aux Waveform')

        self.fft_calc() # Calculate the original FFT

    def _fft_calc(self,t,v):
        t_unscaled = U.time_scale(self.t, self.tunits, 'sec') # Reverse the conversion.
        v_unscaled = U.voltage_scale(self.v, self.vunits, 'v')

        if self.use_step_spectrum:
            f,V = step_spectrum(t_unscaled, v_unscaled)
        else:
            f,V = F.singlesidedspectrum(t, v)
        f_scaled = U.frequency_scale(f, 'Hz', self.funits)
        V_scaled = V # TODO: think about scaling and phase
        return f_scaled, V_scaled

    def fft_calc(self):
        self.f, self.V = self._fft_calc(self.t, self.v)


class WindowedVoltage(VoltageWaveformAndSpectrum):
    """Class to represent a voltage waveform with a ...
    Windowed units (in twin) are expected to be in the final time units
    specfied by the optional kwarg tunits."""
    def __init__(self, t, v, twin, **kwargs):
        VoltageWaveformAndSpectrum.__init__(self, t, v, **kwargs)

        # If twin is a list/tuple of lists/tuples, than accessing the first
        # index should reveal another tuple or list type.  If it is not, then
        # the type should be something other than a list or tuple.
        if type(twin[0]) not in (tuple,list):
            self.twin = [twin,]
        else:
            self.twin = twin
        print( len(self.twin) )

        self.vwin = [None,]*len(self.twin)
        self.Vwin = [None,]*len(self.twin)
        self._recalculate(window_index=None) # Calculates all windows and spectra.

    def add_window(self,twin,index=None):
        """This adds an additional window and adds a slot in the Vwin spectrum
        list for another.."""
        if index is None:
            self.twin.append(twin)
            self.vwin.append(None) # Make empty initially
            self.Vwin.append(None) # Make empty initially
            index = -1 # last index
        else:
            self.twin[index] = twin # Might want to expect exceptions here.

        # Next window the voltage and calculate the spectrum for this new
        # windowed data.
        self._recalculate(window_index=index)

    def rm_window(self,index):
        """Pop a window and its spectrum from the member attribute lists."""
        self.twin.pop(index)
        self.vwin.pop(index)
        self.Vwin.pop(index)

    def _window(self, twin):
        v_windowed = F.zero_outside_window(self.t, self.v, twin, voutside=self.voutside)
        return v_windowed

    def _recalculate(self, window_index=None):
        if window_index is None:
            for ind in range(len(self.twin)):
                self.vwin[ind] = self._window(self.twin[ind])
        else:
            self.vwin[ind] = self._window(self.twin[window_index])
        self.fft_calc_win(window_index=window_index)

    def fft_calc_win(self, window_index=None):
        if window_index is None:
            for ind in range(len(self.twin)):
                #print ind
                #f_tmp, Vwin = self.fft_calc_win(ind)
                self.fft_calc_win(ind)
        else:
            f_tmp, Vwin = self._fft_calc(self.t, self.vwin[window_index]) #
            self.Vwin[window_index] = Vwin

class ExtendingWindowedVoltage(WindowedVoltage):
    """
    An ExtendingWindowsVoltage is slightly different from the WindowedVoltage
    in that the windows all start at one time and the end of the window is the
    only changing aspect.  However, the rectangles in the time domain are drawn
    as if there are slices.
    """
    def __init__(self, t, v, twin=None, steps=50, **kwargs):
        if twin is None:
            tmin = t[0]
            tmax = t[-1]
        else:
            tmin, tmax = twin
        tsteps = np.linspace(tmin,tmax,steps+1)
        twin_l = [(tmin, ts_i) for ts_i in tsteps[1:]]

        WindowedVoltage.__init__(self, t, v, twin_l)


################################################################################

class TimeAndSpectrumView(object):
    """
    TimeAndSpectrumView accepts a WindowedVoltage type along with kwargs for
    setting up the plotting parameters and builds a plot of the time and
    frequency domain data.
    """

    def __init__(self, windowed_voltage, **kwargs):
        # Default attribute values
        self.freq_lim = None
        self.time_lim = None
        self.Vmag_lim = None
        self.v_lim = None
        self.v_aux_lim = None

        self.multifigure = True
        self.title = None
        self.spectrum_title = None
        self.time_response_title = None

        self.freq_label = None
        self.time_label = None
        self.Vmag_label = None
        self.v_label = None

        self.figsize = (8,6)
        self.fig_d = {}
        self.ax_d = dict(time_ax=None, aux_time_ax=None,
                         spectrum_ax=None)

        self.windowed_voltage = windowed_voltage

        set_class_attributes(self,**kwargs)

        self.__build_colormap() # builds the function for color selection

        self.draw_figures() # Might not have this every time.
        self.show_figures()

    def draw_figures(self):
        if self.multifigure:
            self.two_figure_setup()
        else:
            self.one_figure_setup()

        # Draw all the plots on the axes.
        self.__draw_time_ax()
        if hasattr(self.windowed_voltage, 'v_aux'):
            self.__draw_aux_time_ax()
        self.__draw_spectrum_ax()

        # Annotated and set limits.
        self._set_limits_and_annotate()

        # Settle some of the post-drawing aspects.
        self._common_figure_post_draw_setup()

    def show_figures(self):
        for fig_i in self.fig_d.values():
            fig_i.show()

    def _common_figure_post_draw_setup(self):
        for ax_i in self.ax_d.values():
            if ax_i is not None:
                ax_i.grid(True)
                #ax_i.legend(loc='best')

    def one_figure_setup(self):
        fig = plt.figure(figsize=self.figsize)

        if hasattr(self.windowed_voltage, 'v_aux'):
            self.ax_d['time_ax'] = fig.add_subplot(3,1,1)
            self.ax_d['aux_time_ax'] = fig.add_subplot(3,1,2,scalex=self.ax_d['time_ax'])
            self.ax_d['spectrum_ax'] = fig.add_subplot(3,1,3)
        else:
            self.ax_d['time_ax'] = fig.add_subplot(2,1,1)
            self.ax_d['spectrum_ax'] = fig.add_subplot(2,1,2)

        self.fig_d['all'] = fig

        #self._common_figure_setup()

    def two_figure_setup(self):
        fig1 = plt.figure(figsize=self.figsize)
        fig2 = plt.figure(figsize=self.figsize)

        if hasattr(self.windowed_voltage, 'v_aux'):
            self.ax_d['time_ax'] = fig1.add_subplot(2,1,1)
            self.ax_d['aux_time_ax'] = fig1.add_subplot(2,1,2,scalex=self.ax_d['time_ax'])
        else:
            self.ax_d['time_ax'] = fig1.add_subplot(1,1,1)
        self.ax_d['spectrum_ax'] = fig2.add_subplot(1,1,1)

        self.fig_d['time'] = fig1
        self.fig_d['spectrum'] = fig2

        #self._common_figure_setup()

    def _set_limits_and_annotate(self):
        self.ax_d['time_ax'].set_xlim(self.time_lim)
        self.ax_d['time_ax'].set_ylim(self.v_lim)
        self.__draw_rectangles_on_time_ax(self.ax_d['time_ax'])
        if self.ax_d['aux_time_ax'] is not None:
            self.ax_d['aux_time_ax'].set_xlim(self.time_lim)
            self.ax_d['aux_time_ax'].set_ylim(self.v_aux_lim)
            self.__draw_rectangles_on_time_ax(self.ax_d['aux_time_ax'])

        self.ax_d['spectrum_ax'].set_xlim(self.freq_lim)
        self.ax_d['spectrum_ax'].set_ylim(self.Vmag_lim)

        for v in self.fig_d.values():
            v.canvas.draw()

    def __draw_time_ax(self):
        ax = self.ax_d['time_ax']
        t = self.windowed_voltage.t
        v = self.windowed_voltage.v
        label = self.windowed_voltage.name
        if label is None:
            label = 'Full Trace'
        ax.plot(t,v,label=label,color='k')

        for ii, vwin_i in enumerate(self.windowed_voltage.vwin):
             ax.plot(t, vwin_i, label='Window %d' % ii, color=self.get_color(ii))

    def __draw_aux_time_ax(self):
        ax = self.ax_d['aux_time_ax']
        t = self.windowed_voltage.t
        v = self.windowed_voltage.v_aux
        label = self.windowed_voltage.aux_name
        ax.plot(t,v,label=label)

    def __draw_spectrum_ax(self):
        ax = self.ax_d['spectrum_ax']
        f = self.windowed_voltage.f
        V = self.windowed_voltage.V
        label = self.windowed_voltage.name
        if label is None:
            label = 'Full Trace'
        ax.plot(f, dB(V), label=label, color='k')

        # Plot the windowed voltage spectra also.
        for ii, Vwin_i in enumerate(self.windowed_voltage.Vwin):
            ax.plot(f, dB(Vwin_i), label='Window %d' % ii, color=self.get_color(ii))

    def __build_colormap(self):
        # Create a color mapping.
        cm = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=len(self.windowed_voltage.twin)-1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        self.get_color = scalarMap.to_rgba # just save the function to this one.

    def __draw_rectangles_on_time_ax(self, ax):
        ymin, ymax = ax.get_ylim()
        for ii, twin_i in enumerate(self.windowed_voltage.twin):
            r1 = Rectangle((twin_i[0],ymin), twin_i[1]-twin_i[0], ymax-ymin)
            ax.add_patch(r1)



class InteractiveWindowView(TimeAndSpectrumView):
    """
    This view should be equivalent to the sliding_fft plotting.
    """
    def __init__(self, windowed_voltage, **kwargs):


        TimeAndSpectrumView.__init__(self, windowed_voltage, **kwargs)
