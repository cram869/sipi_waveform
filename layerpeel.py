#!/usr/bin/env python3
"""
A layer peeling code for TDR responses.

original author: Michael Cracraft (macracra@us.ibm.com)
"""

def layerpeel(t, v_measured, v_matched = None, Zo_system = 50):

    # Initial parameter checks.
    if len(t) != len(v_measured):
        print("Time and voltage vectors have different numbers of points.")
        return None

    if v_matched is None:
        v_matched = [v_measured[0],] * len(v_measured)
    elif type(v_matched) not in (list, tuple):
        v_matched_const = v_matched
        v_matched = [v_matched_const,] * len(v_measured)

    if len(t) != len(v_matched):
        print("Time and matched voltage vectors have different numbers of points.")

    # Find the left-side from the matched and measured voltages.
    vpL = v_matched[:] # positive going wave, left side
    vnL = [v_measured[i] - v_matched[i] for i in range(len(v_measured))] # negative going wave, left side

    # Propagate the left-side values to the right side.
    vpR = vpL[:]

    vnR = vnL[:]

    # Perform the layer-peeling.
    Zo = [Zo_system,] * len(v_measured)
    for i in range(0, len(v_measured)-1):
        # Step 1: propagate
        if i > 0:
            for j in range(0, len(v_measured)-1):
                vpR[j+1] = vpL[j]
                if j > 0:
                    vnR[j-1] = vnL[j];

        # Step 2: compute next characteristic impedance
        if i > 0:
            Zo[i+1] = Zo[i] * (vpR[i] + vnR[i])/(vpR[i]-vnR[i])

        # Step 3: update the left side for the next step
        for j in range(0, len(v_measured)):
            vpL[j] = vpR[j]*(1+Zo[i+1]/Zo[i])/2. + vnR[j]*(1-Zo[i+1]/Zo[i])/2.
            vnL[j] = vpR[j]*(1-Zo[i+1]/Zo[i])/2. + vnR[j]*(1+Zo[i+1]/Zo[i])/2.

    # Zo_peeled = Zo[:len(v_measured)/2] # just clarifying the name.

    # Raw TDR calculation
    Zo_raw = [Zo_system * (v_measured[i]/(2.*v_matched[i] - v_measured[i]))
              for i in range(0, len(v_measured), 1)] # take only every other step to match lengths.

    return t, Zo, Zo_raw
    #return t[:len(v_measured)/2], Zo_peeled, Zo_raw

def mylayerpeel():
    pass
