#!/usr/bin/env python3
"""
Read and manipulate wfm files and some other oscilloscope and TDR formats.
Includes several Tektronix formats and the HDF5 Keysight format.

original author: Michael Cracraft (macracra@us.ibm.com)
author: Michael Cracraft (cracraft@rose-hulman.edu)

"""
import sys
import os
import os.path
import re
import numpy as np
import struct
from pandas import read_csv as pd_read_csv

from .tracemath import nearest_index
from .util import print_timing

###############################################################################

@print_timing
def readfile(filename, **kwargs):
    """
    I started building a better script for reading general wfm files.
    """

    # Load the file.
    fobj = None
    fileStr = None
    try:
        fobj = open(filename, 'r')
        fileStr = fobj.read()
        fobj.close()
    except:
        print("File opening error.")
        return None

    if fobj is None:
        print("File object is null after opening attempt.")
        return None

    headerString, dataString = fileStr.split('Data:')

    headerDict = readheader(headerString)
    print( headerDict )
    x,y = readdata(dataString, headerDict)
    return x,y

headerpat1 = """Format: (?P<format>.*?)
Type: (?P<type>.*?)
"""
# TODO: Split these patterns up and check for them one at a time. Not all are
# required, and not all will be needed.
format_pat = re.compile('format:(.*)', re.IGNORECASE)
type_pat = re.compile('type:(.*)', re.IGNORECASE)
datatype_pat = re.compile('data type:(.*)', re.IGNORECASE)
sample_pat = re.compile('samples:(.*)', re.IGNORECASE)

number_w_units = '\s*(?P<number>[0-9]*[0-9.][0-9]*[eE]?-?[0-9]*)\s*(?P<unit>[a-zA-Z]*)'
stepsize_pat = re.compile('step size:' + number_w_units, re.IGNORECASE)
firststep_pat = re.compile('first step:(.*)', re.IGNORECASE)
timestep_pat = re.compile('time step:' + number_w_units, re.IGNORECASE)
delay_pat = re.compile('delay:' + number_w_units, re.IGNORECASE)

sheaderpat1 = re.compile(headerpat1, re.M+re.S)
headerpat2time = 'Begin Header.*' + '\r?\n+'.join(["Type: (?P<unittype>.*?)",
                                                   "Samples: (?P<samples>.*?)",
                                                   "Step Size: (?P<stepsize>.*?)",
                                                   "First Step: (?P<firststep>.*?)"]) + '\r?\n'
sheaderpat2time = re.compile(headerpat2time, re.M+re.S)
headerpat2freq = 'Begin Header.*' + '\r?\n+'.join(["Type: (?P<unittype>.*?)",
                                                   "Data Type: (?P<datatype>.*?)",
                                                   "Samples: (?P<samples>.*?)",
                                                   "Step Size: (?P<stepsize>.*?)",
                                                   "First Step: (?P<firststep>.*?)"]) + '\r?\n'
sheaderpat2freq = re.compile(headerpat2freq, re.M+re.S)

snumwithunits = re.compile('\s*(?P<number>[0-9]*[0-9.][0-9]*[eE]?-?[0-9]*)\s*(?P<unit>[a-zA-Z]*)')

def readheader(headerString):
    # Match the first pattern.
    m = sheaderpat1.search(headerString)
    if m:
        d = dict([(key,val.strip().lower()) for key,val in m.groupdict().items()])
    else:
        print( "Did not match the first header string." )
        print( headerString )
        return

    spat = sheaderpat2time
    if 'frequency' in d['type']:
        spat = sheaderpat2freq

    m = spat.search(headerString)
    if not m:
        print( "Did not match the second header string." )
        return
    for key, val in m.groupdict().items():
        d[key] = val.strip().lower()

    # Modify some key header information.
    # Step size
    stepsizestr = d['stepsize']
    m = snumwithunits.search(stepsizestr)
    if m:
        dm = m.groupdict()
        Step = float(dm['number'])
        UnitStr = dm['unit']
    try:
        stepsize = {'s': lambda x: x,
                    'n': lambda x: x * 1.e-9,
                    'p': lambda x: x * 1.e-12,
                    'f': lambda x: x * 1.e-15,
                    'meg': lambda x: x * 1.e6}[UnitStr](Step)
        d['stepsize'] = stepsize
    except KeyError:
        print( '(Step = %g, UnitStr = "%s")' % (Step, UnitStr) )

    # First step
    d['firststep'] = float(d['firststep'])

    # Samples
    d['samples'] = int(d['samples'])

    return d

def readdata(dataString, headerDict):
    d = headerDict
    # Turn the data into a table and then a 2D array.
    DataLines = dataString.strip().splitlines()
    data = []
    for iline in DataLines:
        data.append([float(s) for s in iline.strip().split()])

    y = ((np.array(data)).squeeze()) # one column for each trace.

    # Take special considerations for frequency domain data.
    print( d['type'] )
    if 'frequency' in d['type']:
        # combine the two columns into one complex value.
        yold = y.copy()
        y = np.zeros((yold.shape[0],yold.shape[1]/2), dtype=np.complex64)
        for ii in range(yold.shape[1]/2):
            y[:,ii] = yold[:,2*ii] + 1j*yold[:,2*ii+1]

    # Make the x vector based on the information previously gathered.

    Ndata = y.shape[0] # get the number of rows (entries in each trace)
    if Ndata != d['samples']:
        print( "Samples specified and data read do not match. Nsamples=%d, Ndata=%d" % (d['samples'], Ndata) )

    toffset = 0.
    x = (np.arange(Ndata) * d['stepsize']) + d['firststep'] - toffset

    return x, y

###############################################################################
###############################################################################


def readdiff(file_p, file_n, return_se = False, path=None, **kwargs):
    """
    Read in two files (assume they have the same time base), return either the
    time, vdiff, and vcomm (or if return_se is True, return time, vp, and vn).

    Optionally a root @path can be specified that will be added to each file
    argument.
    """
    if path:
        path_p = os.path.join(path, file_p)
        path_n = os.path.join(path, file_n)
    else:
        path_p = file_p
        path_n = file_n

    tp, vp = readfile(path_p, **kwargs)
    tn, vn = readfile(path_n, **kwargs)

    if return_se:
        return tp, vp, vn
    else:
        vdiff = vp-vn
        vcomm = (vp+vn)/2.
        return tp, vdiff, vcomm

###############################################################################
###############################################################################

# Read an IConnect wfm file, and put it in a two column array format.

# Break into Samples, Time Step, Delay, and Data
def oldreadfile(filename, **kwargs):

    toffset = 0.
    if 'toffset' in kwargs:
        toffset = kwargs['toffset']

    fobj = None
    fileStr = None
    try:
        fobj = open(filename, 'r')
        fileStr = fobj.read().lower()
        fobj.close()
    except:
        print( "File opening error." )
        return None

    if fobj is None:
        return None

    # Find type and find


    # Split at samples.
    SamplesArray = fileStr.split('samples:')
    Nsamples = None
    if len(SamplesArray) < 2:
        print( "No samples found" )
        #return None
    else:
        AfterSamples = SamplesArray[1]
        AfterSamplesArray = AfterSamples.split()
        Nsamples = None
        if len(AfterSamplesArray) < 2:
            print( "No samples found (2)" )
            #return None
        else:
            Nsamples = int(AfterSamplesArray[0])

    # Split at time step.
    TimeStepArray = fileStr.split('step size:')
    if len(TimeStepArray) < 2:
        TimeStepArray = fileStr.split('time step:')
        if len(TimeStepArray) < 2:
            print( "No time step found" )
            #return None

    Tstep = None
    if len(TimeStepArray) >= 2:
        AfterTimeStep = TimeStepArray[1]
        #print( AfterTimeStep )
        pat1 = re.compile('\s*([0-9]*[0-9.][0-9]*[eE]?-?[0-9]*)\s*([a-z]*)')
        mat1 = pat1.search(AfterTimeStep)
        #mat1 = re.search(pat1, AfterTimeStep)
        if mat1 is None:
            print( "No step size found (2)" )
            #return None
        else:
            #print( mat1.groups() )
            BaseTstep = float(mat1.groups()[0])
            UnitMatch = mat1.groups()[1]
            try:
                Tstep = {'s': lambda x: x,\
                         'n': lambda x: x * 1.e-9,\
                         'p': lambda x: x * 1.e-12,\
                         'f': lambda x: x * 1.e-15,\
                         'meg': lambda x: x * 1.e6}[UnitMatch](BaseTstep)
            except KeyError:
                print( '(BaseTstep = %g, UnitMatch = "%s")' % (BaseTstep, UnitMatch) )
    print( Tstep )

    # Split at delay.
    Tdelay = 0.
    DelayArray = fileStr.split('first step:')
    if len(DelayArray) < 2:
        print( "No delay found" )
        #return None
    else:
        AfterDelay = DelayArray[1]
        AfterDelayArray = AfterDelay.split()
        if len(AfterDelayArray) < 2:
            print( "No delay found (2)" )
            #return None
        else:
            Tdelay = float(AfterDelayArray[0])

    # Split at data
    DataStrArray = fileStr.split('data:')
    if len(DataStrArray) < 2:
        print( "No data found" )
        return None

    # Turn the data into a table and then a 2D array.
    DataLines = DataStrArray[1].strip().splitlines()
    data = []
    for iline in DataLines:
        data.append([float(s) for s in iline.strip().split()])

    y = ((np.array(data)).squeeze()) # one column for each trace.

    # Make the x vector based on the information previously gathered.

    Ndata = y.shape[0] # get the number of rows (entries in each trace)
    if Nsamples and Ndata != Nsamples:
        print( "Samples specified and data read do not match. Nsamples=%d, Ndata=%d" % (Nsamples, Ndata) )

    x = (np.arange(Ndata) * Tstep) + Tdelay - toffset

    return x,y

################################################################################
@print_timing
def tekcsvread(filename = None):
    """
    Read the csv data from the Tek Scope. Get col 4 and 5 only.
    This does not close the file on return but returns the file
    object to be reused later to continue loading for the previous
    location.
    """
    rowlist = open(filename, 'r').read().splitlines()
    t = [float(row.split(',')[3]) for row in rowlist]
    v = [float(row.split(',')[4]) for row in rowlist]

    return np.array(t), np.array(v)


################################################################################
@print_timing
def tek_mdo_csvread(filename = None):
    """Example file:
Model,MDO4014B-3
Firmware Version,3.22

Waveform Type,ANALOG
Point Format,Y
Horizontal Units,s
Horizontal Scale,4e-05
Horizontal Delay,0
Sample Interval,4e-10
Record Length,2e+07
Gating,0.0% to 100.0%
Probe Attenuation,10
Vertical Units,V
Vertical Offset,0
Vertical Scale,0.5
Vertical Position,-3.76
,
,
,
Label,
TIME,CH2
-4.0000000e-03,3.34
-3.9999996e-03,3.3
-3.9999992e-03,3.28
-3.9999988e-03,3.3"""

    # Load the file.
    with open(filename, 'r') as fobj:
        line_ = fobj.readline().lower()
        # header_pair_list = []
        # The header information seems unnecessary for basic plotting.
        # Discard it until the 'label' line.
        while 'label' not in line_ :
            # header_pair_list.append(line_.split(DELIMITER))
            line_ = fobj.readline().lower()
        # The last line read should have "label" in it.
        # Throw that line away and get the next one as the data headers.
        # Use the Pandas read_csv routine to do the rest of the work.
        df = pd_read_csv(fobj)
        output_d = {}
        for col_i in df.columns:
            output_d.setdefault(col_i, np.array(df[col_i]))

        return output_d


################################################################################
@print_timing
def xycsvread(filename):
    s = open(filename, 'r').read().strip().splitlines()
    xyStringPairs = [sline.split(',') for sline in s]
    x = np.array([float(xyStringPair[0]) for xyStringPair in xyStringPairs])
    y = np.array([float(xyStringPair[1]) for xyStringPair in xyStringPairs])
    return x,y


################################################################################
@print_timing
def tek_sparameter_wizard_read(path):
    """* Time Domain Analysis Systems:  IConnect
* Version 1.1.2  (Internal Release)

* File Created: 7/8/2015 2:22:07 PM
* Created By: IConnect S-parameter Wizard

* Format: Waveform
* Type: Voltage
* Samples: 1000
* Time Step: 5.00E-12s
* Delay: 0s


* Data:
 0.254145205020905
 0.254160344600677
 0.25422477722168
"""
    def str2number(s_):
        # Convert step size to a real number.
        m = snumwithunits.search(s_)
        if m:
            dm = m.groupdict()
            Step = float(dm['number'])
            UnitStr = dm['unit']
        try:
            stepsize = {'s': lambda x: x,
                        'n': lambda x: x * 1.e-9,
                        'p': lambda x: x * 1.e-12,
                        'f': lambda x: x * 1.e-15,
                        'meg': lambda x: x * 1.e6}[UnitStr](Step)
        except KeyError:
            print( '(Step = %g, UnitStr = "%s")' % (Step, UnitStr) )
        return stepsize


    # Load the file.
    with open(path, 'r') as fobj:
        s_raw = fobj.read().lower()
    header_s = s_raw.split('data:')[0].strip().replace('*','')
    d_ = dict([( s_.split(':',1)[0].strip(), s_.split(':',1)[1].strip() ) for s_ in header_s.splitlines() if ':' in s_])

    d_['stepsize'] = str2number(d_['time step'])

    # First step
    d_['firststep'] = str2number(d_['delay'])

    # Samples
    d_['samples'] = int(d_['samples'])

    # Data
    d_['v'] = np.array([float(s_) for s_ in s_raw.split('data:')[1].strip().split()])

    # Time vector
    d_['t'] = np.arange(0,d_['samples']) * d_['stepsize'] + d_['firststep']

    return d_


##############################################################################

@print_timing
def wfmbinaryreader(filename, **kwargs):

    options = {'nstep': 1,
               'debuglevel': 0}

    for k, v in kwargs.items():
        options[k] = v

    f = open(filename, 'rb')

    def u(fmt, b):
        if 's' in fmt:
            return str(struct.unpack(fmt, f.read(b)))
        else:
            return (struct.unpack(fmt, f.read(b))[0])

    def get_datatype(n):
        datatypes_tmp = zip(['int16', 'int32', 'uint32', 'uint64',
                             'fp32', 'fp64', 'uint8', 'int8', 'invalid'],
                            [2, 4, 4, 8, 4, 8, 1, 1, None],
                            ['h', 'l', 'L', 'Q', 'f', 'd', 'B', 'b'])

        datatypes = [dict(s=s, numbytes=b, unpack_code=c)
                     for s, b, c in datatypes_tmp]
        # l = ['int16', 'int32', 'uint32', 'uint64',
        #     'fp32', 'fp64', 'uint8', 'int8', 'invalid']
        if n >= len(datatypes):
            return datatypes[-1]
        else:
            return datatypes[n]

    def explicit_storagetypes(n_):
        l_ = ['sample', 'min_max', 'vert_hist', 'hor_hist',
              'row_order', 'column_order', 'invalid']
        if n_ >= len(l_):
            return '%s(%d)' % (l_[-1], n_)
        else:
            return l_[n_]

    loglist = []

    def outputline(*arg, **kwargs):
        debuglevel = 0
        if 'debuglevel' in kwargs:
            debuglevel = kwargs['debuglevel']

        # print arg
        s = (''.join([repr(a) for a in arg])).replace("'", "")
        loglist.append({'string': s, 'debuglevel': debuglevel})

    def printlog(**kwargs):
        debuglevel = 0
        if 'debuglevel' in kwargs:
            debuglevel = kwargs['debuglevel']

        L = [d['string'] for d in loglist if d['debuglevel'] <= debuglevel]
        return '\n'.join(L)


    outputline("byte order verification: ", u('2s', 2), debuglevel=2)
    outputline("version number: ", u('8s', 8), debuglevel=2)
    outputline("num digits in byte count: ", u('b', 1), debuglevel=2)
    outputline("number of bytes to EOF: ", u('i', 4), debuglevel=2)
    outputline("number of bytes per point: ", u('b', 1), debuglevel=2)

    nn = f.tell()
    ncurvebufferoffset = u('i', 4)
    outputline("byte offset to beginning of curve buffer: ",
               ncurvebufferoffset, " (", nn, ")", debuglevel=0)

    outputline("horizontal zoom scale factor: ", u('i', 4), debuglevel=2)
    outputline("horizontal zoom position: ", u('f', 4), debuglevel=2)
    outputline("vertical zoom scale factor: ", u('d', 8), debuglevel=2)
    outputline("vertical zoom position: ", u('f', 4), debuglevel=2)
    outputline("waveform label: ", u('32s', 32), debuglevel=2)
    outputline("N (number of FastFrames - 1): ", u('I', 4), debuglevel=2)
    outputline("Size of the waveform header: ", u('H', 2), debuglevel=2)

    # waveform header
    outputline("Waveform Header *****", debuglevel=2)
    s = {0: 'Single waveform',
         1: 'FastFrame'}[u('i', 4)]
    outputline("SetType: ", s, debuglevel=2)
    outputline("WfmCnt: ", u('I', 4), debuglevel=2)
    outputline("Acq Counter: ", u('L', 8), debuglevel=2)
    outputline("Transaction counter: ", u('L', 8), debuglevel=2)
    outputline("Slot ID: ", u('i', 4), debuglevel=2)
    outputline("Is static flag: ", u('i', 4), debuglevel=2)
    outputline("Wfm update specification count: ", u('I', 4), debuglevel=2)
    outputline("Imp dim ref count: ", u('I', 4), debuglevel=2)
    outputline("Exp dim ref count: ", u('I', 4), debuglevel=2)
    s = {0: 'wfmdata_scalar_meas', 1: 'wfmdata_scalar_const',
         2: 'wfmdata_vector', 4: 'wfmdata_invalid',
         5: 'wfmdata_wfmdb'}[u('i', 4)]
    outputline("Data type: ", s, debuglevel=2)
    outputline("Gen purpose counter: ", u('L', 8), debuglevel=2)
    outputline("Accum waveform count: ", u('I', 4), debuglevel=2)
    outputline("Target accum count: ", u('I', 4), debuglevel=2)
    outputline("Curve ref count: ", u('I', 4), debuglevel=2)
    outputline("Num of requested fast frames: ", u('I', 4), debuglevel=2)
    outputline("Num of acquired fast frames: ", u('I', 4), debuglevel=2)
    s = {0: 'summary_frame_off', # Adds 2 bytes to offsets hereafter.
         1: 'summary_frame_average',
         2: 'summary_frame_envelope'}[u('H', 2)]
    outputline("Summary frame type: ", s, debuglevel=2)
    outputline("Pix map display format: ", u('i', 4), debuglevel=2)
    outputline("Pix map max value: ", u('L', 8), debuglevel=2)

    # Explicit dimension 1
    outputline("Explicit dimension 1 ************", debuglevel=2)
    f.seek(168)  # This is two bytes + from the 166 the spec gives.
    nn = f.tell()
    vscale = u('d', 8)
    outputline("Dim scale (voltage?): ", vscale, " (", nn, ")", debuglevel=0)
    nn = f.tell()
    voff = u('d', 8)
    outputline("Dim offset (voltage?): ", voff, " (", nn, ")", debuglevel=0)
    outputline("Dim size: ", u('I', 4), debuglevel=2)
    outputline("Units: ", u('20s', 20), debuglevel=2)
    outputline("Dim extent min: ", u('d', 8), debuglevel=2)
    outputline("Dim extent max: ", u('d', 8), debuglevel=2)
    outputline("Dim extent resolution: ", u('d', 8), debuglevel=2)
    outputline("Dim extent ref pt: ", u('d', 8), debuglevel=2)
    nn = f.tell()
    explicit_datatype1 = get_datatype(u('i', 4))
    outputline("Explicit Dim1 Format: ", explicit_datatype1['s'], " (", nn, ")", debuglevel=0)
    outputline("Storage type: ", explicit_storagetypes(u('i', 4)), debuglevel=2)
    outputline("N value: ", u('i', 4), debuglevel=2)
    outputline("Over range: ", u('i', 4), debuglevel=2)
    outputline("Under range: ", u('i', 4), debuglevel=2)
    outputline("High range: ", u('i', 4), debuglevel=2)
    outputline("Low range: ", u('i', 4), debuglevel=2)
    outputline("User scale: ", u('d', 8), debuglevel=2)
    outputline("User units: ", u('20s', 20), debuglevel=2)
    outputline("User offset: ", u('d', 8), debuglevel=2)
    outputline("Point density: ", u('d', 8), debuglevel=2) # was 4 byte at some point before
    outputline("HRef: ", u('d', 8), debuglevel=2)
    outputline("TrigDelay: ", u('d', 8), debuglevel=2)

    outputline("Explicit dimension 2 ************", debuglevel=2)
    f.seek(328)  # 322 the spec gives.
    outputline("Dim scale (voltage?): ", u('d', 8), debuglevel=2)
    outputline("Dim offset (voltage?): ", u('d', 8), debuglevel=2)
    outputline("Dim size: ", u('I', 4), debuglevel=2)
    outputline("Units: ", u('20s', 20), debuglevel=2)
    outputline("Dim extent min: ", u('d', 8), debuglevel=2)
    outputline("Dim extent max: ", u('d', 8), debuglevel=2)
    outputline("Dim extent resolution: ", u('d', 8), debuglevel=2)
    outputline("Dim extent ref pt: ", u('d', 8), debuglevel=2)
    nn = f.tell()
    explicit_datatype2 = get_datatype(u('i', 4))
    outputline("Explicit Dim2 Format: ", explicit_datatype2['s'], " (", nn, ")", debuglevel=2)
    outputline("Storage type: ", explicit_storagetypes(u('i', 4)), debuglevel=2)
    outputline("N value: ", u('i', 4), debuglevel=2)
    outputline("Over range: ", u('i', 4), debuglevel=2)
    outputline("Under range: ", u('i', 4), debuglevel=2)
    outputline("High range: ", u('i', 4), debuglevel=2)
    outputline("Low range: ", u('i', 4), debuglevel=2)
    outputline("User scale: ", u('d', 8), debuglevel=2)
    outputline("User units: ", u('20s', 20), debuglevel=2)
    outputline("User offset: ", u('d', 8), debuglevel=2)
    outputline("Point density: ", u('d', 8), debuglevel=2)
    outputline("HRef: ", u('d', 8), debuglevel=2)
    outputline("TrigDelay: ", u('d', 8), debuglevel=2)

    outputline("Implicit dimension 1 ************", debuglevel=2)
    f.seek(488)  # 478 the spec gives.
    nn = f.tell()
    tscale = u('d', 8)
    outputline("Dim scale: ", tscale, " (", nn, ")", debuglevel=0)
    nn = f.tell()
    toff = u('d', 8)
    outputline("Dim offset: ", toff, " (", nn, ")", debuglevel=0)
    nn = f.tell()
    npoints = u('I', 4)
    outputline("Dim size: ", npoints, " (", nn, ")", debuglevel=0)
    outputline("Units: ", u('20s', 20), debuglevel=2)
    outputline("Dim extent min: ", u('d', 8), debuglevel=2)
    outputline("Dim extent max: ", u('d', 8), debuglevel=2)
    outputline("Dim extent resolution: ", u('d', 8), debuglevel=2)
    outputline("Dim extent ref pt: ", u('d', 8), debuglevel=2)
    outputline("Spacing: ", u('I', 4), debuglevel=2)
    outputline("User scale: ", u('d', 8), debuglevel=2)
    outputline("User units: ", u('20s', 20), debuglevel=2)
    outputline("User offset: ", u('d', 8), debuglevel=2)
    outputline("Point density: ", u('d', 8), debuglevel=2)
    outputline("HRef: ", u('d', 8), debuglevel=2)
    outputline("TrigDelay: ", u('d', 8), debuglevel=2)

    outputline("Implicit dimension 2 ************", debuglevel=2)
    f.seek(624)
    outputline("Dim scale: ", u('d', 8), debuglevel=2)
    outputline("Dim offset: ", u('d', 8), debuglevel=2)
    outputline("Dim size: ", u('I', 4), debuglevel=2)
    outputline("Units: ", u('20s', 20), debuglevel=2)
    outputline("Dim extent min: ", u('d', 8), debuglevel=2)
    outputline("Dim extent max: ", u('d', 8), debuglevel=2)
    outputline("Dim extent resolution: ", u('d', 8), debuglevel=2)
    outputline("Dim extent ref pt: ", u('d', 8), debuglevel=2)
    outputline("Spacing: ", u('I', 4), debuglevel=2)
    outputline("User scale: ", u('d', 8), debuglevel=2)
    outputline("User units: ", u('20s', 20), debuglevel=2)
    outputline("User offset: ", u('d', 8), debuglevel=2)
    outputline("Point density: ", u('d', 8), debuglevel=2)
    outputline("HRef: ", u('d', 8), debuglevel=2)
    outputline("TrigDelay: ", u('d', 8), debuglevel=2)

    outputline("Time base 1 **********", debuglevel=2)
    f.seek(760)
    outputline("Real point spacing: ", u('I', 4), debuglevel=2)
    s = {0: 'sweep_roll', 1: 'sweep_sample',
         2: 'sweep_et', 3: 'sweep_invalid'}[u('i', 4)]
    outputline("Sweep: ", s, debuglevel=2)
    s = {0: 'base_time', 1: 'base_spectral_mag',
         2: 'base_spectral_phase', 3: 'base_invalid'}[u('i', 4)]
    outputline("Type of base: ", s, debuglevel=2)

    outputline("Time base 2 **********", debuglevel=2)
    f.seek(772)
    outputline("Real point spacing: ", u('I', 4), debuglevel=2)
    s = {0: 'sweep_roll', 1: 'sweep_sample',
         2: 'sweep_et', 3: 'sweep_invalid'}[u('i', 4)]
    outputline("Sweep: ", s, debuglevel=2)
    s = {0: 'base_time',
         1: 'base_spectral_mag',
         2: 'base_spectral_phase',
         3: 'base_invalid'}[u('i', 4)]
    outputline("Type of base: ", s, debuglevel=2)

    outputline("Wfm Update specification ********", debuglevel=2)
    f.seek(784)
    outputline("Real point offset: ", u('I', 4), debuglevel=2)
    outputline("TT offset: ", u('d', 8), debuglevel=2)
    outputline("Frac sec: ", u('d', 8), debuglevel=2)
    outputline("Gmt sec: ", u('i', 4), debuglevel=2)

    outputline("Wfm Curve Information ***********", debuglevel=2)
    f.seek(808)
    outputline("State flags: ", u('I', 4), debuglevel=2)
    outputline("Type of check sum: ", u('i', 4), debuglevel=2)
    outputline("Check sum: ", u('h', 2), debuglevel=2)
    outputline("Precharge start offset: ", u('I', 4), debuglevel=2)
    outputline("Data start offset: ", u('I', 4), debuglevel=2)
    outputline("Postcharge start offset: ", u('I', 4), debuglevel=2)
    outputline("Postcharge stop offset: ", u('I', 4), debuglevel=2)
    outputline("End of curve buffer offset: ", u('I', 4), debuglevel=2)

    outputline("FastFrame Frames ************", debuglevel=2)
    outputline("Not messing with these. :)", debuglevel=2)

    outputline("Reading Curve Buffer ***********", debuglevel=2)

    # nstep = 2500 # just to keep the size of the arrays to a decent level.
    # indexlist_tmp = np.arange(0,npoints,options['nstep'])

    # xmin = options.get('tmin', toff)
    # xmax = options.get('tmax', indexlist_tmp[-1]*tscale + toff)

    # imin = np.floor((xmin - toff)/tscale)
    # imax = np.ceil((xmax - toff)/tscale)

    # index_limits = np.searchsorted(indexlist_tmp, [imin, imax])
    # indexlist = indexlist_tmp[index_limits[0]:index_limits[1]]

    # x = indexlist * tscale + toff
    x_full = (np.arange(0, npoints, options['nstep']) * tscale) + toff

    # If a min and max value were given in the GUI, find the indices
    # to use a subset of the points.  This approach is inefficient in
    # that I should be able to determine the exact indices I need based
    # on the toff and tscale information.  The above lines seem to
    # indicate that I tried and failed since they are commented out.
    imin = 0
    xmin = options.get('tmin', None)
    if xmin is not None:
        imin = nearest_index(x_full, xmin)

    imax = x_full.shape[0]
    xmax = options.get('tmax', None)
    if xmax is not None:
        imax = nearest_index(x_full, xmax)

    x = x_full[imin:imax]

    # Offset to data that I got from one of the reads above. This procedure is
    # not generalized to accept different types of data in the curve buffer
    # yet. Take the 'b' in the unpack below.  That should adjust according to
    # the actual type specified in the header.
    f.seek(ncurvebufferoffset)  # 838
    s = f.read(npoints*explicit_datatype1['numbytes'])
    xx = np.array(struct.unpack('%d%s' % (npoints, explicit_datatype1['unpack_code']), s))  # This method is ~6x faster than the list method below.
    xx_nstep = xx[::options['nstep']]
    y = voff + vscale*xx_nstep[imin:imax]

    del(s)  # take a load off the memory. :)

    outputline("Markers *************", debuglevel = 2)
    outputline("Not messing with these either at the moment. :)",
              debuglevel = 2)

    print( printlog(debuglevel=options['debuglevel']) )

    return x, y


################################################################################

def read_hist(filename):
    fobj = open(filename)
    s_l = [s.strip() for s in fobj.read().strip().splitlines()]
    data = [r.split(',') for r in s_l]
    x_l = map(lambda r: float(r[0]), data)
    counts_l =  map(lambda r: int(r[1]), data)
    return x_l, counts_l

def read_meas(filename):
    pass

################################################################################
################################################################################
################################################################################
#
# from PyQt4 import QtGui
#
# class WfmInterface(QtGui.QMainWindow):
#     def __init__(self, startdir='.'):
#         super( type(self), self).__init__() # Start the QtGui.QMainWindow init.
#
#         self.filedialog_last_path_accessed='.'
#
#         self.initUI()
#
#     def initUI(self):
#         pass
#
#     def useFileDialog(self):
#         fname = QtGui.QFileDialog.getOpenFileName(self, 'Get File Path',
#                                                   self.filedialog_last_path_accessed)
#         # Convert to relative path.
#         p_out = os.path.normpath(unicode(fname))
#         self.filedialog_last_path_accessed = p_out
#         rp_out = os.path.relpath(p_out)
#         return rp_out

################################################################################
################################################################################
################################################################################
# Python 3 only
import h5py
def keysight_hdf5_load(filename, channel='Channel 1'):
    o = h5py.File(filename, 'r')
    filetype = o['FileType']['KeysightH5FileType'][()]
    print('FileType: ', filetype, '\n')
    frame = o['Frame']['TheFrame'][()]
    print('Frame: ', frame, '\n')
    wfm = o['Waveforms']
    ch = wfm[channel] # Probably should try to generalize this.

    chattrs = ch.attrs
    print(channel, ' Attributes:\n', chattrs.keys())

    num_points = chattrs['NumPoints']

    xorg = chattrs['XOrg']
    xinc = chattrs['XInc']
    xunits = chattrs['XUnits']
    x_array = np.arange(num_points)*xinc + xorg

    yorg = chattrs['YOrg']
    yinc = chattrs['YInc']
    yunits = chattrs['YUnits']

    channel_data_key_str = channel + 'Data'
    chdata = ch[channel_data_key_str]
    yint = chdata[:]
    y_array = yint*yinc + yorg

    return x_array, y_array





################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    #import sys
    arglist = sys.argv[1:]
    if len(arglist) > 0:
        print( readfile(arglist[0]) )
