import numpy as np
import lqcdjk_fitting as fit
import functions as fncs
import mpi_functions as mpi_fncs
from mpi4py import MPI

# E = sqrt( m^2 + p^2 )

def energy( mEff, pSq, L ):

    return np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * pSq )


# Q^2 = (p_f - p_i)^2 - (E_f - E_i)^2

def calcQsq( p_fin, q_list, mEff, L, mpi_info ):

    # p_fin[ p ]
    # q_list[ q ]
    # m[ b ]
    # L

    binNum = len( mEff )
    
    Qsq_list = [ [] for b in mEff ]
    Qsq_where = [ [] for b in mEff ]

    for m, ib in fncs.zipXandIndex( mEff ):

        # Qsq_p_q[ p, q ]

        Qsq_p_q = np.zeros( ( len( p_fin ), len( q_list ) ) )

        for p, ip in fncs.zipXandIndex( p_fin ):
            for q, iq in fncs.zipXandIndex( q_list ):

                p_ini = p - q
                
                Qsq_p_q[ ip, iq ] \
                    = ( 2. * np.pi / L ) ** 2 \
                    * np.dot( p - p_ini, p - p_ini ) \
                    - ( energy( m, np.dot( p, p ), L )
                        - energy( m, np.dot( p_ini, p_ini ), L ) ) ** 2

        Qsq_list[ ib ] = np.sort( np.unique( Qsq_p_q ) )

        Qsq_where[ ib ] = [ [] for qs in Qsq_list[ ib ] ]

        for qs, iqs in fncs.zipXandIndex( Qsq_list[ ib ] ):

            Qsq_where[ ib ][ iqs ] = Qsq_p_q == qs

    Qsq_list = np.array( Qsq_list )
    Qsq_where = np.array( Qsq_where )

    QsqNum = Qsq_list.shape[ -1 ]

    # Check that Q^2's are at the same place across bins

    for ib in range( 1, binNum ):
        
        if not np.array_equal( Qsq_where[ ib ],
                               Qsq_where[ 0 ] ):
            
            error = "Error (physQuants.fourVectorQsq): " \
                    "Qsq_where[ {} ] != Qsq_where[ 0 ]" 
        
            mpi_fncs.mpiPrint( Qsq_where[ ib ],
                               mpi_info )
            mpi_fncs.mpiPrint( Qsq_where[ 0 ],
                               mpi_info )

            mpi_fncs.mpiPrintError( error.format( ib ),
                                    mpi_info )

    if mpi_info[ 'binNum_loc' ]:

        Qsq_where = Qsq_where[ 0 ]

    return Qsq_list, QsqNum, Qsq_where


# KK = sqrt( 2E ( E + m ) )

def KK_nucleon( mEff, Qsq, L ):

    return np.sqrt( 2.0 * energy( mEff, Qsq, L ) \
                    * ( energy( mEff, Qsq, L ) \
                        + mEff ) )


# KK = C_1^-1 = 2 sqrt( EE' )

def KK_meson( mEff, pSq_ini, pSq_fin, L ):

    return 2.0 * np.sqrt( energy( mEff, pSq_ini, L )
                          * energy( mEff, pSq_fin, L ) )


def twopFit( c0, E0, t ):
#def twopFit( c0, c1, E0, E1, t ):

    return c0 * np.exp( -E0 * t )
    #return c0 * np.exp( -E0 * t ) + c1 * np.exp( -E1 * t )


# Calculate the kinematic factor based on the decomposition
# of the given form factor

def kineFactor( ratio_err, formFactor, particle, flavor,
                mEff, p_fin, Q, L, mpi_info ):

    if formFactor == "GE_GM":

        kineFactor = kineFactor_GE_GM( ratio_err, particle, flavor,
                                       mEff, p_fin, Q, L,
                                       mpi_info )

    elif formFactor == "BT10":

        kineFactor = kineFactor_BT10( ratio_err, particle, flavor,
                                      mEff, p_fin, Q, L,
                                      mpi_info )

    elif formFactor == "FS":

        kineFactor = kineFactor_FS( ratio_err, particle, flavor,
                                      mEff, p_fin, Q, L,
                                      mpi_info )

    elif formFactor == "A20_B20":

        kineFactor = kineFactor_A20_B20( ratio_err, particle, flavor,
                                         mEff, p_fin, Q, L,
                                         mpi_info )

    elif formFactor == "A30_B30":

        kineFactor = kineFactor_A30_B30( ratio_err, particle, flavor,
                                         mEff, p_fin, Q, L,
                                         mpi_info )

    elif formFactor == "A40_B40_C40":

        kineFactor = kineFactor_A40_B40_C40( ratio_err, particle, flavor,
                                         mEff, p_fin, Q, L,
                                         mpi_info )

    return kineFactor


# Calculate the kinematic factor based on the vector form factor
# decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_GE_GM( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                      mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                  p_fin.shape[ 0 ] ),
                                mpi_info )

    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                  Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [GE,GM] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                if particle == "nucleon":

                    Qsq = np.dot( q, q )
                    
                    kineFactor[ b, ip, iq ] \
                        = [ [ ( energy( mEff[ b ],
                                        Qsq, L ) \
                                + mEff[ b ] ), 0 ],
                            [ -2.0 * np.pi / L * q[ 0 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 1 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 2 ], 0 ],
                            [ 0, -2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 0 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 0 ] ] ] \
                        / np.repeat( ratio_err[ ip, iq ] ** 2,
                                     2).reshape( ratioNum, 2 ) \
                        / KK_nucleon( mEff[ b ], 0, Qsq, L )
                    
                else: # particle == "meson"

                    p_ini = p - q

                    pSq_ini = np.dot( p_ini, p_ini )
                    pSq_fin = np.dot( p, p )

                    kineFactor[ b, ip, iq ] \
                        = [ [ energy( mEff[ b ],
                                      pSq_ini,
                                      L )
                              + energy( mEff[ b ],
                                        pSq_fin,
                                        L ),
                              0 ], 
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 0 ] + p[ 0 ] ),
                              0 ],
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 1 ] + p[ 1 ] ),
                              0 ],
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 2 ] + p[ 2 ] ),
                              0 ] ] \
                        / np.repeat( ratio_err[ ip, iq ] ** 2,
                                     2).reshape( ratioNum, 2 ) \
                        / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                                            
                # End if meson                                   
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculate the kinematic factor based on the tensor form
# factor decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_BT10( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                     mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A20_B20): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [BT10,0] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            
            pSq_fin = np.dot( p, p )

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                p_ini = p - q
                    
                qSq = np.dot( q, q )
                pSq_ini = np.dot( p_ini, p_ini )

                E_ini = energy( mEff[ b ], pSq_ini, L )
                E_fin = energy( mEff[ b ], pSq_fin, L )

                kineFactor[ b, ip, iq ] \
                    = [ [ ( 2. * np.pi / L ) ** 2
                          * ( p_ini[ 0 ] * p[ 1 ]
                            - p_ini[ 1 ] * p[ 0 ] )
                          / mEff[ b ],
                          0 ],
                        [ ( 2. * np.pi / L ) ** 2
                          * ( p_ini[ 0 ] * p[ 2 ]
                            - p_ini[ 2 ] * p[ 0 ] )
                          / mEff[ b ],
                          0 ],
                        [ ( 2. * np.pi / L ) ** 2
                          * ( p_ini[ 1 ] * p[ 2 ]
                            - p_ini[ 2 ] * p[ 1 ] )
                          / mEff[ b ],
                          0 ],
                        [ 2. * np.pi / L
                          * ( E_fin * p_ini[ 0 ] - E_ini * p[ 0 ] )
                          / mEff[ b ],
                          0 ],
                        [ 2. * np.pi / L
                          * ( E_fin * p_ini[ 1 ] - E_ini * p[ 1 ] )
                          / mEff[ b ],
                          0 ],
                        [ 2. * np.pi / L
                          * ( E_fin * p_ini[ 2 ] - E_ini * p[ 2 ] )
                          / mEff[ b ],
                          0 ] ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 2).reshape( ratioNum, 2 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculate the kinematic factor based on the scalar
# form factor decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_FS( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                   mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A20_B20): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [FS,0] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            
            pSq_fin = np.dot( p, p )

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                pSq_ini = np.dot( p - q, p - q )

                kineFactor[ b, ip, iq ] \
                    = [ 1., 0. ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 2).reshape( ratioNum, 2 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculate the kinematic factor based on the 1-derivative
# form factor decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_A20_B20( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                        mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A20_B20): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [A20,B20] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            
            pSq_fin = np.dot( p, p )

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                p_ini = p - q
                    
                qSq = np.dot( q, q )
                pSq_ini = np.dot( p_ini, p_ini )

                E_ini = energy( mEff[ b ], pSq_ini, L )
                E_fin = energy( mEff[ b ], pSq_fin, L )

                #CJL:HERE

                kineFactor[ b, ip, iq ] \
                    = [ [ 1./4. * ( mEff[ b ] ** 2
                                    - 2. * ( E_fin + E_ini ) ** 2
                                    + E_fin * E_ini
                                    - ( 2. * np.pi / L ) ** 2
                                    * ( p[ 0 ] * p_ini[ 0 ]
                                        + p[ 1 ] * p_ini[ 1 ]
                                        + p[ 2 ] * p_ini[ 2 ] ) ),
                          -( mEff[ b ] ** 2
                             - 2 * ( E_fin - E_ini ) ** 2
                             - E_fin * E_ini
                             + ( 2. * np.pi / L ) ** 2
                             * ( p[ 0 ] * p_ini[ 0 ]
                                 + p[ 1 ] * p_ini[ 1 ]
                                 + p[ 2 ] * p_ini[ 2 ] ) ) ],
                        [ 1./2. * ( E_fin + E_ini )
                          * 2. * np.pi / L
                          * ( p[ 0 ] + p_ini[ 0 ] ),
                          2. * ( E_fin - E_ini )
                             * 2. * np.pi / L
                             * ( p[ 0 ] - p_ini[ 0 ] ) ],
                        [ 1./2. * ( E_fin + E_ini )
                          * 2. * np.pi / L
                          * ( p[ 1 ] + p_ini[ 1 ] ),
                          2. * ( E_fin - E_ini )
                          * 2. * np.pi / L
                          * ( p[ 1 ] - p_ini[ 1 ] ) ],
                        [ 1./2. * ( E_fin + E_ini )
                          * 2. * np.pi / L
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( E_fin - E_ini )
                          * 2. * np.pi / L
                             * ( p[ 2 ] - p_ini[ 2 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 1 ] + p_ini[ 1 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] - p_ini[ 0 ] )
                          * ( p[ 1 ] - p_ini[ 1 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] - p_ini[ 0 ] )
                          * ( p[ 2 ] - p_ini[ 2 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] + p_ini[ 1 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] - p_ini[ 1 ] )
                          * ( p[ 2 ] - p_ini[ 2 ] ) ] ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 2).reshape( ratioNum, 2 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculate the kinematic factor based on the 2-derivative
# form factor decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_A30_B30( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                        mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A30_B30): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A30_B30): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A30_B30): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [A30,B30] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                p_ini = p - q
                    
                qSq = np.dot( q, q )

                pSq_ini = np.dot( p_ini, p_ini )
                pSq_fin = np.dot( p, p )

                E_ini = energy( mEff[ b ], pSq_ini, L )
                E_fin = energy( mEff[ b ], pSq_fin, L )

                kineFactor[ b, ip, iq ] \
                    = [ [ 1./4. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 1 ] + p_ini[ 1 ] ),
                          1./3. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] * p_ini[ 1 ] + p[ 1 ] * p_ini[ 0 ] )
                          + ( 1./3. * E_ini - E_fin )
                          * ( 2. * np.pi / L ) ** 2
                          * p[ 0 ] * p[ 1 ]
                          + ( 1./3. * E_fin - E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * p_ini[ 0 ] * p_ini[ 1 ] ],
                        [ 1./4. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          1./3. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] * p_ini[ 2 ] + p[ 2 ] * p_ini[ 0 ] )
                          + ( 1./3. * E_ini - E_fin )
                          * ( 2. * np.pi / L ) ** 2
                          * p[ 0 ] * p[ 2 ]
                          + ( 1./3. * E_fin - E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * p_ini[ 0 ] * p_ini[ 2 ] ],
                        [ 1./4. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] + p_ini[ 1 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          1./3. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] * p_ini[ 2 ] + p[ 2 ] * p_ini[ 1 ] )
                          + ( 1./3. * E_ini - E_fin )
                          * ( 2. * np.pi / L ) ** 2
                          * p[ 1 ] * p[ 2 ]
                          + ( 1./3. * E_fin - E_ini )
                          * ( 2. * np.pi / L ) ** 2
                          * p_ini[ 1 ] * p_ini[ 2 ] ] ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 2).reshape( ratioNum, 2 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
            
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculate the kinematic factor based on the 3-derivative
# form factor decomposition

# ratio_err[ p, Q, r ]
# "particle"
# "flavor"
# mEff[ b ]
# p_fin[ p, pi ]
# momList[ Q, qi ]
# L

def kineFactor_A40_B40_C40( ratio_err, particle, flavor,
                            mEff, p_fin, Q, L,
                            mpi_info ):

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A40_B40_C40): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A40_B40_C40): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A40_B40_C40): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [ A40, B40, C40 ] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 3 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                p_ini = p - q
                    
                qSq = np.dot( q, q )

                pSq_ini = np.dot( p_ini, p_ini )
                pSq_fin = np.dot( p, p )

                E_ini = energy( mEff[ b ], pSq_ini, L )
                E_fin = energy( mEff[ b ], pSq_fin, L )

                kineFactor[ b, ip, iq ] \
                    = [ [ -1./8. * ( E_fin + E_ini )
                          * ( 2. * np.pi / L ) ** 3
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 1 ] + p_ini[ 1 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          E_fin * ( 2. * np.pi / L ) ** 3
                          * ( 1./6. * p[ 2 ]
                              * p_ini[ 0 ] * p_ini[ 2 ]
                              + 1./6. * p[ 1 ]
                              * p_ini[ 0 ] * p_ini[ 2 ]
                              + 1./6. * p[ 0 ]
                              * p_ini[ 1 ] * p_ini[ 2 ]
                              - 1./2. * p[ 0 ]
                              * p[ 1 ] * p[ 2 ] )
                          + E_ini * ( 2. * np.pi / L ) ** 3
                          * ( 1./6. * p[ 1 ] * p[ 2 ]
                                    * p_ini[ 0 ]
                                    + 1./6. * p[ 0 ] * p[ 2 ]
                                    * p_ini[ 1 ]
                                    + 1./6. * p[ 0 ] * p[ 1 ]
                                    * p_ini[ 2 ]
                                    - 1./2. * p_ini[ 0 ]
                                    * p_ini[ 1 ] * p_ini[ 2 ] ),
                          ( 2. * np.pi / L ) ** 3
                          * ( ( 1./2. * E_ini - E_fin )
                              * p[ 0 ] * p[ 1 ] * p[ 2 ]
                              + ( E_ini - 1./2. * E_fin )
                              * p_ini[ 0 ] * p_ini[ 1 ] * p_ini[ 2 ]
                              + E_fin * ( 1./2. * p[ 1 ]
                                          * p[ 2 ] * p_ini[ 0 ]
                                          + 1./2. * p[ 0 ]
                                          * p[ 2 ] * p_ini[ 1 ]
                                          + 1./2. * p[ 0 ]
                                          * p[ 1 ] * p_ini[ 2 ] )
                              - E_ini * ( 1./2. * p[ 2 ]
                                          * p_ini[ 0 ] * p_ini[ 1 ]
                                          + 1./2. * p[ 0 ]
                                          * p_ini[ 1 ] * p_ini[ 2 ]
                                          + 1./2. * p[ 1 ]
                                          * p_ini[ 0 ]
                                          * p_ini[ 2 ] ) ) ] ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 3 ).reshape( ratioNum, 3 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
            
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


# Calculates form factors using singular value decomposition

# kineFactor_loc[ b_loc, p, q, ratio, [ F1, F2 ] ]
# ratio[ b, p, q, ratio ]
# ratio_err[ p, q, ratio ]
# mpi_info
    
def calcFormFactors_SVD( kineFactor_loc, ratio, ratio_err, Qsq_where,
                         formFactor, pSq, mpi_info ):

    # Set MPI variables

    comm = mpi_info[ 'comm' ]

    binNum = mpi_info[ 'binNum_glob' ]
    binNum_loc = mpi_info[ 'binNum_loc' ]

    binList_loc = mpi_info[ 'binList_loc' ]

    recvCount = mpi_info[ 'recvCount' ]
    recvOffset = mpi_info[ 'recvOffset' ]

    # Set dimension sizes

    qNum = kineFactor_loc.shape[ 2 ]

    QsqNum = len( Qsq_where )
    ratioNum = kineFactor_loc.shape[ -2 ]

    formFactorNum = kineFactor_loc.shape[ -1 ]

    # Local ratios on this process

    ratio_loc = ratio[ binList_loc ]

    # Gather kinematic factors
    # kineFactor[ b, p, q, r, [ F1, F2 ] ]

    kineFactor = np.zeros( ( binNum, ) + kineFactor_loc.shape[ 1: ] )
    
    comm.Allgatherv( kineFactor_loc,
                     [ kineFactor,
                       recvCount \
                       * np.prod( kineFactor_loc.shape[ 1: ] ),
                       recvOffset \
                       * np.prod( kineFactor_loc.shape[ 1: ] ),
                       MPI.DOUBLE ] )

    # Repeat error for each bin

    ratio_err_loc = np.array( [ ratio_err ] * binNum_loc )
    ratio_err_loc = ratio_err_loc.reshape( ( binNum_loc, )
                                           + ratio.shape[ 1: ] )

    ratio_err_glob = np.array( [ ratio_err ] * binNum )
    ratio_err_glob = ratio_err_glob.reshape( ratio.shape )

    # Initialize local form factors
    # F_loc[ b_loc, qs, [ F1, F2 ] ]

    F_loc = np.zeros( ( binNum_loc, QsqNum, formFactorNum ), dtype=float )
    
    # Loop over Q^2
    for iqs in range( QsqNum ):

        if not np.any( Qsq_where[ iqs ] ):

            continue

        # Get kinematic factors for this Q^2
        # kineFactor_Qsq[ b_loc, Q^2[ qs ], r, [ F1, F2 ] ]
        
        kineFactor_Qsq = kineFactor_loc[ :, Qsq_where[ iqs ], :, : ]

        # Get ratios and their errors for this Q^2
        # ratio_Qsq[ b_loc, Q^2[ qs ], r ]

        ratio_Qsq = ratio_loc[ :, Qsq_where[ iqs ], : ]
        ratio_err_Qsq = ratio_err_loc[ :, Qsq_where[ iqs ], : ]
            
        # Number of combinations of p and q
        # for this value of Q^2

        QsqNum_Qsq = kineFactor_Qsq.shape[ 1 ]

        # Combine Q^2 and ratio dimensions, i.e., 
        # kineFactor_Qsq[ b_loc, Q^2[ qs ], r, [ F1, F2 ] ]
        # -> kineFactor_Qsq[ b_loc, Q^2[ qs ] * r, [ F1, F2 ] ]

        kineFactor_Qsq = kineFactor_Qsq.reshape( binNum_loc,
                                                 QsqNum_Qsq * ratioNum,
                                                 formFactorNum )

        # ratio_Qsq[ b, Q^2[ qs ], r ]
        # -> ratio_Qsq[ b, Q^2[ qs ] * r ]

        ratio_Qsq = ratio_Qsq.reshape( binNum_loc,
                                       QsqNum_Qsq * ratioNum )
        ratio_err_Qsq \
            = ratio_err_Qsq.reshape( binNum_loc, QsqNum_Qsq * ratioNum )

        # Perform SVD
        # kineFactor_Qsq = u s v^T

        u, s, vT = np.linalg.svd( kineFactor_Qsq, full_matrices=False )

        # Calculate kineFactor_Qsq^-1 = ( v s^-1 u^T )^T
                
        uT = np.transpose( u, ( 0, 2, 1 ) )
        v = np.transpose( vT, ( 0, 2, 1 ) )
            
        # s is array of values on diagonal, make a diagonal matrix
        # with appropriate shape for matrix multiplication and
        # calculate its inverse

        smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
        smat_inv = np.zeros( ( binNum_loc, )
                             + np.transpose( smat ).shape )
    
        # Loop over bins
        for b in range( binNum_loc ):
                    
            smat[ :vT.shape[ -2 ], :vT.shape[ -2 ] ] = np.diag( s[ b ] )
                    
            smat_inv[ b ] = np.linalg.pinv( smat )
    
        # End loop over bins

        # Multiply matrices to get decomop = kineFactor_Qsq^-1
        # decomp[ b_loc, Q^2[qs]*ratio, [ F1, F2 ] ]
            
        decomp = np.transpose( v @ smat_inv @ uT, ( 0, 2, 1 ) )

        # Axes to sum over

        sum_axes = tuple( range( 1, ratio_Qsq.ndim ) )
 
        # Loop over form factors
        for iff in range( formFactorNum ):

            # Calculate form factor for this Q^2

            F_loc[ :, iqs, iff ] \
                = np.sum( decomp[ ..., iff ]
                          * ratio_Qsq
                          / ratio_err_Qsq ** 2,
                          axis=sum_axes )                
                              
        # End loop over form factors

    # End loop over Q^2
    
    return F_loc


def calcFormFactors_ti( kineFactor, ratio, ratio_fit_err,
                        Qsq_where, iQsq, mpi_info ):

    # kineFactor[ b, p, q, r, ff ]
    # ratio[ b, p, q, r, t ]
    # ratio_fit_err[ p, q, r ]
    # Qsq_where[ qs ]
    # iQsq

    kineFactor_Qsq = kineFactor[ :, Qsq_where[ iQsq ], :, :]
    ratio_Qsq = ratio[ :, Qsq_where[ iQsq ], :, : ]
    ratio_fit_err_Qsq = ratio_fit_err[ Qsq_where[ iQsq ], : ]

    binNum = kineFactor_Qsq.shape[ 0 ]
    QsqNum_Qsq = kineFactor_Qsq.shape[ 1 ]
    ratioNum = kineFactor_Qsq.shape[ 2 ]
    formFactorNum = kineFactor_Qsq.shape[ 3 ]
    T = ratio.shape[ -1 ]

    # Combine Q^2 and ratio dimensions

    # kineFactor_Qsq[ b, Q^2[ qs ], r, [ F1, F2 ] ]
    # -> kineFactor_Qsq[ b, Q^2[ qs ] * r, [ F1, F2 ] ]

    kineFactor_Qsq = kineFactor_Qsq.reshape( binNum, QsqNum_Qsq * ratioNum,
                                             formFactorNum )

    # ratio_Qsq[ b, Q^2[ qs ], r, t ]
    # -> ratio_Qsq[ b, Q^2[ qs ] * r, t ]

    ratio_Qsq = ratio_Qsq.reshape( binNum, QsqNum_Qsq * ratioNum, T )

    # ratio_Qsq[ Q^2[ qs ], r ]
    # -> ratio_Qsq[ Q^2[ qs ] * r ]

    ratio_fit_err_Qsq = ratio_fit_err_Qsq.reshape( QsqNum_Qsq * ratioNum )

    F = np.zeros( ( binNum, formFactorNum, T ) )

    # Loop over form factors:
    for iff in range( formFactorNum ):
        # Loop over time
        for it in range( T ):

            F_sum = 0.
            F_count = 0

            # Loop over Q^2 and ratio
            for iqr in range( QsqNum_Qsq * ratioNum ):

                if not np.any( kineFactor_Qsq[ :, iqr, iff ] == 0 ):

                    F_sum += ratio_Qsq[ :, iqr, it ] \
                             / kineFactor_Qsq[ :, iqr, iff ] \
                             / ratio_fit_err_Qsq[ iqr ] ** 2
                    
                    F_count += 1

            # End loop over Q^2 and ratio
            
            if F_count == 0:

                continue

            F[ :, iff, it ] = F_sum / F_count

        # End loop over time
    # End loop over form factors
    
    return F

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


# avgXKineFactor = 2E/(1/2*m^2-2*E^2)

def avgXKineFactor( mEff, momSq, L ):

    return 2.0 * energy( mEff, momSq, L ) \
        / ( 0.5 * mEff ** 2 \
            - 2.0 * energy( mEff, momSq, L ) ** 2 )


def avgX2KineFactor( mEff, momSq, L ):

    return -energy( mEff, momSq, L ) / mEff


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


def calcAvgX2( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX2 = np.zeros( threep.shape )

    preFactor = avgX2KineFactor( mEff, momSq, L )

    for t in range( threep.shape[ 1 ] ):
           
        avgX2[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX2


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

    preFactor = avgXKineFactor( mEff, momSq, L )

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


def calcMatrixElemEM_ratio( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    ratio = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        ratio[ :, t ] = threep[ :, t ] / twop_tsink

    return ratio


def calcMatrixElemEM_twopFit( threep, tsink, c0, E0 ):
#def calcMatrixElemEM_twopFit( threep, tsink, c0, c1, E0, E1 ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    #c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    #E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    ratio = threep / twopFit( c0_cp, E0_cp, tsink )
    #ratio = threep / twopFit( c0_cp, c1_cp, E0_cp, E1_cp, tsink )

    return ratio


def calcMellin_twopFit( threep, tsink, mEff, momSq, L, \
                        c0, E0, moment, mpi_info ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( threep.shape )
    E0_cp = np.repeat( E0, T ).reshape( threep.shape )

    if moment == 1 or moment == "avgX":

        preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                               T ).reshape( threep.shape )

    elif moment == 2 or moment == "avgX2":

        preFactor = -1.0

    elif moment == 3 or moment == "avgX3":

        preFactor = -1.0        

    #mpi_fncs.mpiPrint(threep[0]/twopFit(c0_cp[0],E0_cp[0],tsink),
    #                  mpi_info)
    #mpi_fncs.mpiPrint(preFactor[0],
    #                  mpi_info)

    return preFactor * threep \
        / twopFit( c0_cp, E0_cp, tsink )


def calcAvgX_twopFit( threep, tsink, mEff, momSq, L, \
                      c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                           T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX


def calcAvgX2_twopFit( threep, tsink, mEff, momSq, L, \
                       c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    #preFactor = np.repeat( avgX2KineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    preFactor = -1.0

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX2 = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX2


def calcAvgX_twopTwoStateFit( threep, tsink, mEff, momSq, L, T, \
                              c0, c1, E0, E1 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # c0[ b ]
    # c1[ b ]
    # E0[ b ]
    # E1[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor=1.0
    #preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / fit.twoStateTwop( tsink, T, \
                               c0_cp, c1_cp, \
                               E0_cp, E1_cp )
    return avgX


def calcAvgX_twopOneStateFit( threep, tsink, mEff, momSq, L, T, G, E ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # G[ b ]
    # E[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor = energy( mEff, momSq, L ) \
                / ( mEff * ( 0.5 * mEff ** 2 \
                             - 2.0 * energy( mEff, momSq, L ) ** 2 ) )

    G_cp = np.repeat( G, T ).reshape( binNum, T )
    E_cp = np.repeat( E, T ).reshape( binNum, T )
    
    avgX = preFactor * threep \
           / fit.oneStateTwop( tsink, T, \
                               G_cp, E_cp )

    return avgX


def calcAvgX_twoStateFit( a00, c0, mEff, momSq, L, ZvD1 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD1 * avgXKineFactor( mEff, momSq, L ) \
        * a00 / c0

def calcAvgX2_twoStateFit( a00, c0, mEff, momSq, L, ZvD2 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD2 * avgX2KineFactor( mEff, momSq, L ) \
        * a00 / c0

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


def calcFormFactorRatio( threep, twop, tsink ):
    
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


def calcFormFactorRatio_twopFit( threep, c0, E0, tsink, p_fin,
                                 Q, pSq_twop, L, dispRelation,
                                 mpi_info ):
    
    # threep[ ..., p, Q, r, t ]
    # c0[ ..., p^2 ]
    # mEff[ ..., p^2 ]
    # tsink
    # p_fin[ p, pi ]
    # Q[ q, qi ]
    # pSq_twop[ ps ]
    # L

    QNum = threep.shape[ -3 ]
    ratioNum = threep.shape[ -2 ]
    
    if QNum != len( Q ):

        error_template = "Error (physQuants.calcFormFactorRatio_twopFit): " \
                         + "length of threep Q dimension {} " \
                         + "does not match length of " \
                         + "Q {}."

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       len( Q ) ),
                                mpi_info )

    ratio = np.zeros( threep.shape[ :-1 ] + ( tsink + 1, ) )

    # Calculate twop from fit parameters
    # twop[ ..., p^2, r, t ]

    twop = np.zeros( c0.shape + ( ratioNum, tsink + 1 ) )

    # Loop over twop p^2
    for ps, ips in fncs.zipXandIndex( pSq_twop ):
        # Loop over time
        for t in range( tsink + 1 ):

            if dispRelation:

                twop_tmp = twopFit( c0[ ..., ips ],
                                    energy( E0, ps, L ),
                                    t )

            else:

                twop_tmp = twopFit( c0[ ..., ips ],
                                    E0[ ..., ips ],
                                    t )

            twop_tmp = np.repeat( twop_tmp, ratioNum )

            twop[ ..., ips, :, t ] \
                = twop_tmp.reshape( twop[ ..., ips,
                                          :, t ].shape )

        # End loop over t
    # End loop over twop p^2

    for p, ip in fncs.zipXandIndex( p_fin ):
        
        pSq_fin = np.dot( p, p )

        pSq_fin_where = np.where( pSq_twop == pSq_fin )

        for q, iq, in fncs.zipXandIndex( Q ):

            pSq_ini = np.dot( p - q, p - q )

            pSq_ini_where = np.where( pSq_twop == pSq_ini )

            for t in range( tsink + 1 ):

                ratio[..., ip, iq, :, t] \
                    = threep[ ..., ip, iq, :, t ] \
                    / twop[ ..., pSq_fin_where, :, tsink ] \
                    * np.sqrt( twop[ ..., pSq_ini_where, :, tsink - t ]
                               * twop[ ..., pSq_fin_where, :, t ]
                               * twop[ ..., pSq_fin_where, :, tsink ]
                               / ( twop[ ..., pSq_fin_where, :, tsink - t ]
                                   * twop[ ..., pSq_ini_where, :, t ]
                                   * twop[ ..., pSq_ini_where, :, tsink ] ) )
            
            # End loop over t
        # End loop ovet Q
    # End loop over p

    return ratio


def calcFormFactorRatio_tsf( a00, c0, pList, qList, pSq_twop, mpi_info ):
    
    # a00[ b, p, q, r ]
    # c0[ b, p^2 ]
    # pList[ p, [ x, y, z ] ]
    # qList[ q, [ x, y, z ] ]
    # pSq_twop[ p^2 ]

    ratio = np.zeros( a00.shape )

    for p, ip in fncs.zipXandIndex( pList ):
        
        pSq_fin = np.dot( p, p )
        
        for q, iq in fncs.zipXandIndex( qList ):

            pSq_ini = np.dot( p - q, p - q )

            for ir in range( a00.shape[ 3 ] ):

                c0_ini = np.squeeze( c0[ :, pSq_twop == pSq_ini ] )
                c0_fin = np.squeeze( c0[ :, pSq_twop == pSq_fin ] )
                
                ratio[ :, ip, iq, ir ] = a00[ :, ip, iq, ir ] \
                                         / np.sqrt( c0_ini * c0_fin )
                
            # End loop over ratio
        # End loop over q
    # End loop over p

    return ratio


# Calculate the electromagnetic form factor.

# threep:

def calcEMFF( threep, twop, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        #factor = 1.0
        factor = 2 * energy / ( energy + mEff )
        #factor = 4.0 * np.sqrt( energy * mEff ) / ( energy + mEff )
        
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

