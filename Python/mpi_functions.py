from sys import stderr
from mpi4py import MPI
import numpy as np

def lqcdjk_mpi_init():

    mpi_info = {}

    mpi_info[ 'comm' ] = MPI.COMM_WORLD
    mpi_info[ 'procNum' ] = mpi_info[ 'comm' ].Get_size()
    mpi_info[ 'rank' ] = mpi_info[ 'comm' ].Get_rank()

    return mpi_info

def lqcdjk_mpi_confs_info( mpi_confs_info ):

    configList = mpi_confs_info[ 'configList' ]

    configNum = len( configList )
    mpi_confs_info[ 'configNum' ] = configNum

    binSize = mpi_confs_info[ 'binSize' ]
    procNum = mpi_confs_info[ 'procNum' ]
    rank = mpi_confs_info[ 'rank' ]

    assert configNum % binSize == 0, \
        "Number of configurations " + str( configNum ) \
        + " not evenly divided by bin size " \
        + str( binSize ) + "."

    #assert configNum % procNum == 0, \
    #    "Number of configurations " + str( configNum ) \
    #    + " not evenly divided by number of processes " \
    #    + str( procNum ) + "."

    # Total number of bins across processes
    binNum_glob = mpi_confs_info[ 'configNum' ] // mpi_confs_info[ 'binSize' ]
    mpi_confs_info[ 'binNum_glob' ] = binNum_glob

    conf_remain = configNum % procNum

    # Number of configurations on each process
    procSize = mpi_confs_info[ 'configNum' ] // mpi_confs_info[ 'procNum' ]


    # Global index of confs for each process
    iconf = [ [] for r in range( procNum ) ]

    for r in range( procNum - conf_remain ):

        #iconf[ r ] = [ r * procSize + cl for cl in range( procSize ) ]
        iconf[ r ] = [ ir for ir  in range( r * procSize, ( r + 1 ) * procSize ) ]

    for rr in range( conf_remain ):

        #iconf[ rr ] = [ r * procSize + cl for cl in range( procSize ) ]
        iconf[ procNum - conf_remain + rr ] \
            = [ ir for ir in range( ( procNum - conf_remain + rr ) 
                                    * procSize + rr, 
                                    ( procNum - conf_remain + rr + 1 ) 
                                    * procSize + 1 + rr ) ]

    #iconf = np.array( [ np.array( [ r * procSize + cl
    #                                for cl in range( procSize ) ], 
    #                              dtype=int )
    #                    for r in range( procNum ) ] )

    # List of global indices for configurations on this process
    configList_loc = [ configList[ ic ] for ic in iconf[ rank ] ]
    mpi_confs_info[ 'configList_loc' ] = configList_loc

    # Global index of first conf of bins for each process
    binStart = np.array( [ np.array( [ cl for cl in iconf[ r ]
                                       if cl % binSize == 0 ], 
                                     dtype=int )
                           for r in range( procNum ) ] )

    # Global bin index for each process
    bin_glob = binStart // binSize

    # List of global bin indices on this process
    binList_loc = bin_glob[ rank ]
    mpi_confs_info[ 'binList_loc' ] = binList_loc

    # Number of bins for each process
    binNum = [ len( binStart[ r ] ) for r in range( procNum ) ]

    # Number of bins for this process
    binNum_loc = binNum[ rank ]
    mpi_confs_info[ 'binNum_loc' ] = np.array( binNum_loc, dtype='i' )

    # Number of configurations on each process and offset 
    # for gatherv functions

    confNum = np.zeros( procNum, dtype=int )
    confOffset = np.zeros( procNum, dtype=int )
    offsetSum = 0

    for r in range( procNum ):

        confNum[ r ] = len( iconf[ r ] )
        confOffset[ r ] = offsetSum

        offsetSum += confNum[ r ]
        
    mpi_confs_info[ 'confNum' ] = confNum
    mpi_confs_info[ 'confOffset' ] = confOffset

    # Number of bins on each process and offset for gatherv functions
    recvCount, recvOffset = recvCountOffset( procNum, binNum )
    mpi_confs_info[ 'recvCount' ] = recvCount
    mpi_confs_info[ 'recvOffset' ] = recvOffset


# Prints message run by first process

# message: Message to be printed
# rank: Rank of process. Prints if rank is 0

def mpiPrint( message, mpi_info ):

    if mpi_info[ 'rank' ] == 0:

        print( message )


# Same as mpiPrint() but prints to stderr and aborts

# message: Message to be printed
# rank: Rank of process. Prints if rank is 0

def mpiPrintErr( message, mpi_info ):

    if mpi_info[ 'comm' ].Get_rank() == 0:

        print( message, file=stderr )
        
        stderr.flush()

    mpi_info[ 'comm' ].Abort()


# Prints message on every process in order

# message: Message to be printed
# mpi_info: MPI info dictionary containing communicator 

def mpiPrintAllRanks( message, mpi_info ):
    
    comm = mpi_info[ 'comm' ]
    rank = mpi_info[ 'rank' ]
    procNum = mpi_info[ 'procNum' ]

    for p in range( procNum ):

        if p == rank:

            print( "rank {}: {}".format( rank, message ) )

        comm.Barrier()


# Return number of bins to be received and their offsets
# for each process. Used for sendv/receivev type operations.

# procNum: Number of processes
# binNum: Number of bins

def recvCountOffset( procNum, binNum ):

    recvCount = [ None for p in range( procNum ) ]
    recvOffset = [ None for p in range( procNum ) ]
    offsetSum = 0

    for p in range( procNum ):

        recvCount[ p ] = binNum[ p ]

        recvOffset[ p ] = offsetSum

        offsetSum += binNum[ p ]

    return np.array( recvCount ), np.array( recvOffset )
