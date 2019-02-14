import numpy as np
import argparse as argp
from scipy.optimize import minimize
import functions as fncs
import readWrite as rw
import physQuants as pq

def polyFit( p2, p4 ):

    # p2[ b, x ]

    # p4[ y ][ b, x ]
    
    fit = []

    # Check that number of bins is the same for all values of tsink

    binNum = []

    yNum = len( p4 )

    for iy in range( yNum ):
 
        binNum.append( p4[ iy ].shape[ 0 ] )

        assert binNum[ iy ] == binNum [ 0 ], "Number of bins not the same for every value of tsink."

    for ib in range( binNum[ 0 ] ):

        a = -10.0
        b = -10.0 
        c = -10.0 
        d = -10.0 

        fitParams = np.array( [ a, b, c, d ] )

        # x[ts][t]

        x = []

        # tsink[ts]

        y = [] 

        p2_cp = []

        p4_cp = []

        for iy in range( yNum ):
 
            xNum = p4[ iy ].shape[ -1 ]

            x.append( np.array( range( xNum ) ) )

            y.append( xNum - 1 )

            #print xNum

            #print x

            #print y

            # p2_cp[ y ]

            p2_cp.append( p2[ ib, y[ -1 ] ] )

            #p4_cp[ y ][ x ]

            p4_cp.append( p4[ iy ][ ib, : ] )

        x = np.array( x )

        y = np.array( y )

        p2_cp = np.array( p2_cp )

        #fit.append( leastsq( twoStateErrorFunction, fitParams, \
        #                     args = ( ti, tsink, p2_cp, p4_cp ) )[0] )

        fit.append( minimize( polyErrorFunction, fitParams, \
                              args = ( x, y, p2_cp, p4_cp ), \
                              method='Powell', jac = False, \
                              options = {'maxiter':100000} ).x )

        #fit.append( differential_evolution( twoStateErrorFunction, bounds, fitParams, \
        #                                    args = ( ti, tsink, p2_cp, p4_cp ) )[0] )
        

    # End loop over bins

    return np.array( fit )


def polyErrorFunction( fitParams, x, y, p2, p4 ):

    a = fitParams[ 0 ]
          
    b = fitParams[ 1 ]
          
    c = fitParams[ 2 ]
          
    d = fitParams[ 3 ]

    # p2Err[ ts ]

    #p2Err = np.array( ( ( twoStateP2( tsink, c0, c1, E0, E1 ) \
    #                        - p2 ) / p2 ) )

    #print "y: " + str(y) + ", a: " + str(a) + ", b: " + str(b)

    #print poly2( y, a, b )

    #print p2

    p2Err = np.array( ( ( poly2( y, a, b ) \
                          - p2 ) / p2 ) ** 2 )

    # p4Err[ ts ][ ti ]

    #p4 *= 10 ** 3

    p4Err = []

    for ix_y, iy, p4_y in zip( x, y, p4 ):

        for ix, p4_x in zip( ix_y, p4_y ):

            #p4Err.append( ( ( twoStateP4( t, ts, a00, a01, a11, E0, E1 ) \
            #                    - p4_ti ) / p4_ti ) )

            #print "x: " + str(ix) + ", y: " + str(iy) + ", a: " + str(a) + ", b: " + str(b) + ", c: " + str(c) + ", d: " + str(d)

            #print "data: " + str(p4_x)

            #print "function: " + str(poly4( ix, iy, a, b, c, d ))

            #print ( ( poly4( ix, iy, a, b, c, d ) \
            #                  - p4_x ) / p4_x ) ** 2 

            p4Err.append( ( ( poly4( ix, iy, a, b, c, d ) \
                              - p4_x ) / p4_x ) ** 2 )

    #return np.concatenate( ( p2Err, p4Err ) )

    return np.sum( np.concatenate( ( p2Err, p4Err ) ) )


def poly4( x, y, a, b, c, d ):

    return a * x + b * x ** 2 + c * y + d * y ** 2

def poly2( y, a, b ):

    return a * y  + b * y ** 2


#########################
# Parse input arguments #
#########################

#########
# Setup #
#########

y = [ 6, 8, 10 ]

yNum = len( y )

x = range( 1, 13 )

xNum = len( x )

binSize = 2

# Check inputs

configNum = 8

binNum = configNum / binSize

######
# p2 #
######

# p2[ c, x ]

p2 = np.random.rand( configNum, xNum )

#print p2

for ic in range( configNum ):

    for ix in range( xNum ):

        p2[ ic, ix ] += poly2( ix, 1, 2 )

        #print p2[ ic, ix ]

        #print poly2( ix, 1, 2 )

#print p2

# Jackknife
# p2_jk[ b, t ]

p2_jk = fncs.jackknife( p2, binSize )

p2_avg = np.average( p2_jk, axis=0 )

p2_err = np.std( p2_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

p2_ts = []

p4_jk = []

p4_avg = []

p4_err = []

for iy in y:
    
    ######
    # p4 #
    ######

    p4 = np.random.rand( configNum, iy+1 )

    #print p4

    for ic in range( configNum ):

        for ix in range( iy+1 ):

            p4[ ic, ix ] += poly4( ix, iy, 1, 2, 3, 4 )

            #print p4[ ic, ix ]

            #print poly4( ix, iy, 1.1, 1.2, 1.3, 1.4 )

    # Jackknife
    # p4_jk[ ts ][ b, x ]
    
    p4_jk.append( fncs.jackknife( p4, binSize ) )

    p4_avg = np.average( p4_jk[ -1 ], axis=0 )

    p4_err = np.std( p4_jk[ -1 ], axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

# End loop over tsink

#print p4_jk

############
# Poly Fit #
############

# fitParams[ b, param ]

fitParams = polyFit( p2_jk, p4_jk )

print fitParams

exit()


a = fitParams[ :, 0 ]
          
b = fitParams[ :, 1 ]

c = fitParams[ :, 2 ]
          
d = fitParams[ :, 3 ]

# Write curve with constant tsink

curve = np.zeros( ( binNum, tsinkNum, 50 ) )

t_i= np.zeros( ( tsinkNum, 50 ) )

for b in range( binNum ):

    for ts in range( tsinkNum ):

        t_i[ ts, : ] = np.linspace( -2, tsink[ ts ] + 2, 50 )

        for t in range( t_i.shape[ -1 ] ):

            curve[ b, ts, t ] = fncs.twoStateThreep( t_i[ ts, t ], tsink[ ts ], \
                                                     a00[ b ], a01[ b ], a11[ b ], \
                                                     E0[ b ], E1[ b ] )

# Average over bins

curve_avg = np.average( curve, axis=0 )

curve_err = np.std( curve, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
                        
fitParams_avg = np.average( fitParams, axis=0 )

fitParams_err = np.std( fitParams, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

#####################
# Write output file #
#####################

for ts in range( tsinkNum ):

    curveOutputFilename = output_template.replace( "*", "threep_twoStateFit_curve_tsink" + str( tsink[ ts ] ) )

    rw.writeAvgDataFile_wX( curveOutputFilename, t_i[ ts ], curve_avg[ ts ], curve_err[ ts ] )

avgXParamsOutputFilename = output_template.replace( "*", "threep_twoStateFitParams" )

rw.writeTSFParamsFile( avgXParamsOutputFilename, fitParams_avg, fitParams_err )

