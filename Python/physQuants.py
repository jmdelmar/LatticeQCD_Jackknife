import numpy as np
import mpi_functions as mpi_fncs

def energy( mEff, Qsq, L ):

    return np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * Qsq )


def KK( mEff, Qsq, L ):

    return np.sqrt( 2.0 * energy( mEff, Qsq, L ) \
                    * ( energy( mEff, Qsq, L ) \
                        + mEff ) )


def kineFactor_GE_GM( ratio_err, mEff, Q, L ):

    # ratio_err[ Q, r ]
    # mEff[ b ]
    # momList[ Q, p ]
  
    momNum = ratio_err.shape[ 0 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    assert len( Q ) == momNum, "Error (kineFactor_GE_GM): " \
        + "momentum dimension of ratio errors " \
        + str( momNum ) + " and momentum transfers " \
        + str( len( Q ) ) + " do not match. "

    # kineFactor[ b, Q, r, [GE,GM] ]

    kineFactor = np.zeros( ( binNum, momNum, ratioNum, 2 ) )

    for b in range( binNum ):

        for q in range( momNum ):

            Qsq = np.dot( Q[ q ], Q[ q ] )

            kineFactor[ b, q ] = [ [ ( energy( mEff[ b ], \
                                               Qsq, L ) \
                                       + mEff[ b ] ), 0 ], \
                                   [ -2.0 * np.pi / L * Q[ q, 0 ], 0 ], \
                                   [ -2.0 * np.pi / L * Q[ q, 1 ], 0 ], \
                                   [ -2.0 * np.pi / L * Q[ q, 2 ], 0 ], \
                                   [ 0, -2.0 * np.pi / L * Q[ q, 2 ] ], \
                                   [ 0, 2.0 * np.pi / L * Q[ q, 1 ] ], \
                                   [ 0, 2.0 * np.pi / L * Q[ q, 2 ] ], \
                                   [ 0, -2.0 * np.pi / L * Q[ q, 0 ] ], \
                                   [ 0, -2.0 * np.pi / L * Q[ q, 1 ] ], \
                                   [ 0, 2.0 * np.pi / L * Q[ q, 0 ] ] ] \
                / np.repeat( ratio_err[ q ], 2).reshape( 10 ,2 ) \
                / KK( mEff[ b ], Qsq, L )

    return kineFactor


def calc_gE_gM( decomp, ratio, ratio_err, Qsq_start, Qsq_end ):

    binNum = decomp.shape[ 0 ]

    gE = np.zeros( ( binNum ) )
    gM = np.zeros( ( binNum ) )

    for b in range( binNum ):

        gE[ b ] = 2.0 * np.sum( decomp[ b, ..., 0 ] \
                                * ratio[ b, \
                                         Qsq_start \
                                         : Qsq_end + 1 ] \
                                / ratio_err[ Qsq_start \
                                             : Qsq_end \
                                             + 1 ] )

        gM[ b ] = 2.0 * np.sum( decomp[ b, ..., 1 ] \
                                * ratio[ b, \
                                         Qsq_start \
                                         : Qsq_end + 1 ] \
                                / ratio_err[ Qsq_start \
                                             : Qsq_end \
                                             + 1 ] )
    
    return gE, gM


def calc_GE_GM( gE, gM, mEff, Qsq, L ):

    # gE[ b ], gM[ b ], mEff[ b ]
    # Qsq, L

    GE = gE \
         + ( energy( mEff, Qsq, L ) - mEff ) \
         / ( energy( mEff, Qsq, L ) + mEff ) * gM

    GM = 2 * mEff / ( energy( mEff, Qsq, L ) + mEff ) \
         * ( gM - gE )

    return GE, GM


# Convert Q^2 from units of (2pi/L)^2 to GeV^2

# Qsq: Q^2 values to be converted
# mEff: Effective mass of particle
# a: Lattice spacing of ensemble
# L: Spacial dimension length of ensemble

def convertQsqToGeV( Qsq, mEff, a, L ):

    Qsq_GeV = 2.0 * ( 0.197 / a ) ** 2 * mEff * ( energy( mEff, Qsq, L ) \
                                                  - mEff )

    return Qsq_GeV


# Calcuate the effective mass from two-point functions which have been
# symmetrized

# twop: Symmetrized two-point functions with last dimension as time

def mEffFromSymTwop( twop ):

    halfT = twop.shape[ -1 ]

    mEff = np.zeros( twop.shape )

    for t in range( 1, halfT - 1 ):

        mEff[ ..., t ] = 1.0 / 2.0 \
                         * np.log( ( twop[ ..., t - 1 ] \
                                     + np.sqrt(twop[ ..., \
                                                     t - 1 ] ** 2 \
                                               - twop[ ..., \
                                                       halfT - 1 ] ** 2) ) \
                                   / ( twop[ ..., t + 1 ] \
                                       + np.sqrt(twop[ ..., \
                                                       t + 1 ] ** 2 \
                                                 - twop[ ..., \
                                                         halfT - 1 ] ** 2) ))

    return mEff


# Calculate the effective mass from two-point functions

# twop: Two-point functions with last dimension as time

def mEff( twop ):

    mEff = np.zeros( twop.shape )

    # Loop through timestep, excluding the last timestep

    for t in range( twop.shape[ -1 ] - 1 ):

        mEff[ ..., t ] = np.log( twop[ ..., t ] / twop[ ..., t + 1 ] )

    # Calculate effective mass at last timestep, 
    # applying boundary conditions

    mEff[ ..., -1 ] = np.log( twop[ ..., -1 ] / twop[ ..., 0 ] )

    return mEff


# Calculate the quark momentum fraction <x> for three-point functions with
# zero final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle

def calcAvgX( threep, twop_tsink, mEff ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):
           
        avgX[ :, t ] = -4.0 / 3.0 / mEff * threep[ :, t ] / twop_tsink

    return avgX


# Calculate the quark momentum fraction <x> for three-point functions with
# finite final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle
# momSq: Final momentum squared
# L: Spacial dimension length of ensemble

def calcAvgX_momBoost( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    # prefactor = 8/3 * E / ( E^2 + p^2 )

    preFactor = -8.0 / 3.0 * energy( mEff, momSq, L ) \
                / ( energy( mEff, momSq, L ) ** 2 \
                    + ( 2.0 * np.pi / L ) ** 2 * momSq )

    #preFactor = 1.0

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


# Calculate the axial charge gA.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink

def calcgA( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    gA = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        gA[ :, t ] = threep[ :, t ] / twop_tsink

    return gA


# Calculate the cosh form of the two-point functions

# t: 
# energy:
# tsink:

def twopCosh( t, energy, tsink ):

    return np.exp( - energy * t ) + np.exp( - energy * ( tsink - t ) )


def calcRatio_Q( threep, twop, tsink ):
    
    # threep[ ..., Q, t ]
    # twop[ ..., Q, t ]

    ratio = np.zeros( threep.shape )
    
    for q in range( threep.shape[ -2 ] ):
        for t in range( threep.shape[ -1 ] ):

            ratio[..., q, t] = threep[ ..., q, t ] / twop[ ..., 0, tsink ] \
                               * np.sqrt( np.abs( twop[ ..., q, tsink - t ] \
                                                  * twop[ ..., 0, t ] \
                                                  * twop[ ..., 0, tsink ] \
                                                  / ( twop[ ..., 0, tsink - t ] \
                                                      * twop[ ..., q, t ] \
                                                      * twop[ ..., q, tsink ] ) ) )  

    return ratio

# Calculate the electromagnetic form factor.

# threep:

def calcEMFF( threep, twop, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        factor = 4.0 * np.sqrt( energy * mEff ) / ( energy + mEff )

        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / twop[ 0, :, tsink ] \
                             * np.sqrt( twop[ q, :, tsink - t ] * twop[ 0, :, t ] * twop[ 0, :, tsink ] \
                                        / ( twop[ 0, :, tsink - t ] * twop[ q, :, t ] * twop[ q, :, tsink ] ) )

    return emff


def calcEMFF_cosh( threep, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        factor = np.sqrt( energy / mEff ) * energy / ( energy + mEff )

        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / threep[ 0, :, t ] \
                              * twopCosh( t, mEff, tsink ) / twopCosh( t, energy, tsink )

    return emff


def calcEMFF_ratio( threep, twop, Qsq, mEff ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        for b in range( threep.shape[ 1 ] ):

            for t in range( threep.shape[ 2 ] ):

                emff[ q, b, t ] = threep[ q, b, t ] / twop[ q, b,  t ] \
                                   / threep[0,b,t] * twop[ 0, b, t ]

    return emff

