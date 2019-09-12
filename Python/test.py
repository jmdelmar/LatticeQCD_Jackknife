import numpy as np

def twoStateTwop( tsink, T, c0, c1, E0, E1 ):

    return c0 * ( np.exp( -E0 * tsink ) \
                  + np.exp( -E0 * ( T - tsink ) ) ) \
        + c1 * ( np.exp( -E1 * tsink ) \
                 + np.exp( -E1 * ( T - tsink ) ) )

ts = np.array( [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] )

twop = np.array( [ 0.00266031,0.00228392,0.00198952,0.00174216,0.00152868,0.00134549,
                   0.00118465,0.0010454,0.00092259,0.00081428,0.00071915,0.00063607,
                   0.000563,0.00049874,0.00044143,0.00039127,0.00034734,0.00030938,
                   0.00027596,0.00024683,0.0002213,0.00019937,0.00018059,0.00016469] )

twop_err = np.array( [  2.04584430e-05, 1.77630556e-05, 1.51594847e-05, 1.23890066e-05,
                        1.04250645e-05, 8.88427213e-06, 7.42966385e-06, 6.42779379e-06,
                        5.39583639e-06, 4.80282607e-06, 4.26188042e-06, 3.73791079e-06,
                        3.33862604e-06, 3.07453824e-06, 2.77451961e-06, 2.58891876e-06,
                        2.40008116e-06, 2.19630610e-06, 2.01495010e-06, 1.76959795e-06,
                        1.62855159e-06, 1.43967489e-06, 1.30507140e-06, 1.22103467e-06] )

c0 = 3.20991746e-03
c1 = 5.99951714e-04
E0 = 1.25153224e-01
E1 = 6.66664326e-01

twop_fit = twoStateTwop( ts, 64, c0, c1, E0, E1 )

res = ( ( twop - twop_fit ) / twop_fit ) ** 2

chiSq = sum( res )

print(twop_fit)

print(res)
print(chiSq)