#!/usr/bin/env python
"""
PRBS Generation Code and Utilities
original author: Michael Cracraft (cracraft@rose-hulman.edu)
"""

taps_d = {
  2: [2, 1], # 2: x^2 + x + 1
  3: [3, 2], # 3: x^3 + x^2 + 1
  4: [4, 3], # 4: x^4 + x^3 + 1
  5: [5, 3], # 5: x^5 + x^3 + 1
  6: [6, 5], # 6: x^6 + x^5 + 1
  7: [7, 6], # 7: x^7 + x^6 + 1
  11: [11, 9], # 11: x^11 + x^9 + 1
  13: [13, 4, 3, 1], # 13: x^13 + x^4 + x^3 + x + 1
  15: [15, 14], # 15: 
  23: [23, 18], # 23: x^23 + x^18 + 1
  32: [32, 22, 2, 1] } # 32: x^32 + x^22 + x^2 + x + 1

# Generate a PRBS bit stream as an array of 0s and 1s.
def prbs_(seed, N):
  taps = taps_d[N]
  bitstream = []
  seed &= 2**N - 1 # ANDing with 0x1111... (N length)
  lfsr = seed
  first_loop = True
  ii = 0
  print(ii, " ", bin(seed), " ", bin(seed & 0b1))
  bitstream.append(seed & 0b1)
  ii += 1
  while ((lfsr != seed) or first_loop) and ii < (2**N + 10):
    xor_term = 0
    for tap_i in taps:
        xor_term ^= lfsr >> (tap_i-1)
    output_bit = xor_term & 0b1 # Provides the feedback operation, shifting the bits from the 4 and 3 position to the first.
    lfsr = ((lfsr<<1) | output_bit) & (2**N-1) # Shift the LFSR one bit to the left and append the new bit to the LSB.  The last AND truncates the bit string to N bits again.
    print(ii, " ", bin(lfsr), " ", bin(output_bit))
    ii += 1
    first_loop = False
    bitstream.append(output_bit)
  return bitstream

prbs7 = lambda seed: prbs_(seed, 7)
prbs13 = lambda seed: prbs_(seed, 13)
prbs23 = lambda seed: prbs_(seed, 23)
prbs32 = lambda seed: prbs_(seed, 32)

def bitstream_complement(bitstream_a):
  return np.ones(bitstream_a.shape, dtype=int) - bitstream_a

