import numpy as np

def convertQsqToGeV( Qsq, mEff, a, L ):

    energy = np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * Qsq )

    Qsq_GeV = 2.0 * ( 0.197 / a ) ** 2 * mEff * ( energy - mEff )

    return Qsq_GeV

def mEffFromSymTwop( twop ):

    halfT = twop.shape[ -1 ]

    mEff = np.zeros( twop.shape )

    for t in range( 1, twop.shape[ -1 ] - 1 ):

        mEff[ ..., t ] = 1.0 / 2 \
                         * np.log( ( twop[ ..., t - 1 ] \
                                     + np.sqrt( twop[ ..., t - 1 ] ** 2 \
                                                - twop[ ..., halfT - 1 ] ** 2 ) ) \
                                   / ( twop[ ..., t + 1 ] \
                                       + np.sqrt( twop[ ..., t + 1 ] ** 2 \
                                                  - twop[ ..., halfT - 1 ] ** 2 ) ) )

    return mEff


def mEff( twop ):

    mEff = np.zeros( twop.shape )

    # Recursively loop though each axis of twop, until the
    # last dimension, which should be time, is reached

    if( twop.ndim > 1 ):

        for dim in range( len( twop ) ):

            mEff[dim] = mEff( twop[dim] )

    else:

        timeLength = len( twop )

        # Loop through timestep, excluding the last timestep

        for t in range( timeLength - 1 ):

            mEff[ t ] = np.log( twop[ t ] / twop[ t + 1 ] )

            # Calculate effective mass at last timestep, 
            # applying boundary conditions

        mEff[ timeLength - 1 ] = np.log( twop[ timeLength - 1 ] / twop[ 0 ] )

    return np.array( mEff )


def calcAvgX( threep, twop_tsink, mEff ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        avgX[ :, t ] = -4.0 / 3.0 / mEff * threep[ :, t ] / twop_tsink

    return avgX


def calcAvgX_momBoost( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    pSq = (2*np.pi/L)**2 * momSq

    energy = np.sqrt( mEff**2 + pSq )
    """
    preFactor = -2.0 / mEff**2 * \
                energy * ( energy + mEff ) \
                / ( 3 * energy**2 + pSq )
    """
    preFactor = 1.0
    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


def calcgA( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    gA = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        gA[ :, t ] = threep[ :, t ] / twop_tsink

    return gA


def twopCosh( t, energy, tsink ):

    return np.exp( - energy * t ) + np.exp( - energy * ( tsink - t ) )


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

