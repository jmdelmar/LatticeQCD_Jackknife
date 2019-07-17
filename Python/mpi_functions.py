from sys import stderr
from mpi4py import MPI
import numpy as np

# Prints message run by first process

# message: Message to be printed
# rank: Rank of process. Prints if rank is 0

def mpiPrint( message, rank ):

    if rank == 0:

        print( message )


# Same as mpiPrint() but prints to stderr

# message: Message to be printed
# rank: Rank of process. Prints if rank is 0

def mpiPrintErr( message, comm ):

    if comm.Get_rank() == 0:

        print( message, file=stderr )
        
        stderr.flush()

    comm.Abort()


# Prints message on every process in order

# message: Message to be printed
# comm: MPI comm object 

def mpiPrintAllRanks( message, comm ):

    rank = comm.Get_rank()

    for p in range( comm.Get_size() ):

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
