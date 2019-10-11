import time
import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq
import lqcdjk_fitting as fit

L = 64
ratioNum = 10

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Perform SVD for GE and GM" )

parser.add_argument( "ratio_err_filename", action='store', type=str )

parser.add_argument( "effective_mass_filename", action='store', type=str )

parser.add_argument( "momentum_list_filename", action='store', type=str )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

args = parser.parse_args()

#########
# Setup #
#########

ratio_err_filename = args.ratio_err_filename

mEff_filename = args.effective_mass_filename

momList_filename = args.momentum_list_filename

output_template = args.output_template

dummy, binNum = rw.detTimestepAndConfigNum( mEff_filename )

##############
# Read files #
##############

# Momentum list

momList = rw.readTxtFile( momList_filename, 3, dtype=int )

momNum = len( momList )

Qsq, Qsq_start, Qsq_end = fncs.processMomList( momList )

QsqNum = len( Qsq )

# Ratio errors
# ratio_err[ Q, r ]

ratio_err = rw.readNthDataCol( ratio_err_filename, \
                               2 ).reshape( ratioNum, \
                                            momNum ).T

# Effective masses
# mEff[ b ]

mEff = rw.readNthDataCol( mEff_filename, 1 )

###############################
# Calculate kinematic factors #
###############################

# kineFacter[ b, Q, r, [ GE, GM ] ]

kineFactor = pq.kineFactor_GE_GM( ratio_err, mEff, momList, L )

###############
# Perform SVD #
###############
"""
u = np.zeros( ( binNum, QsqNum * ratioNum, 2 ) )
s = np.zeros( ( binNum, QsqNum, 2 ) )
vT = np.zeros( ( binNum, QsqNum, 2, 2) )
"""
inverse = [ [] for qsq in range( QsqNum ) ]

for qsq in range( QsqNum ):

    kineFactor_Qsq \
        = kineFactor[ :, \
                      Qsq_start[ qsq ]:Qsq_end[ qsq ] + 1, \
                      ... ].reshape( binNum, \
                                     ( Qsq_start[ qsq ] \
                                       - Qsq_end[ qsq ] + 1 ) \
                                     * ratioNum, 2 )
    """
    dummy_u, s[ :, qsq ], vT[ :, qsq ] = np.linalg.svd( kineFactor_Qsq, full_matrices=False )
    """
    u, s, vT = np.linalg.svd( kineFactor_Qsq, full_matrices=False )

    ##########################
    # Calculate ( s^-1 v )^T #
    ##########################

    uT = np.transpose( u, ( 0, 2, 1 ) )
    v = np.transpose( vT, ( 0, 2, 1 ) )

    smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
    smat_inv = np.zeros( ( binNum, ) + np.transpose( smat ).shape )

    for b in range( binNum ):

        smat[ :vT.shape[ -2 ], :vT.shape[ -2 ] ] = np.diag( s[ b ] )

        smat_inv[ b ] = np.linalg.pinv( smat )

    inv_tmp =  v @ smat_inv @ uT

    inverse[ qsq ] = np.average( inv_tmp, axis = 0 ).T

################
# Write output #
################

output_filename = output_template.replace( "*", "SVD_output" )

rw.writeSVDOutputFile( output_filename, inverse, Qsq )
