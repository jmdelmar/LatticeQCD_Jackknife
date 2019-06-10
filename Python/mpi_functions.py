from sys import stderr
from mpi4py import MPI
import numpy as np


def mpiPrint( message, rank ):

    if rank == 0:

        print( message )


def mpiPrintErr( message, rank ):

    if rank == 0:

        print( message, file=stderr )


def mpiPrintAllRanks( message, comm ):

    rank = comm.Get_rank()

    for p in range( comm.Get_size() ):

        if p == rank:

            print( "rank {}: {}".format( rank, message ) )

        comm.Barrier()

def recvCountOffset( procNum, binNum ):

    recvCount = [ None for p in range( procNum ) ]
    recvOffset = [ None for p in range( procNum ) ]
    offsetSum = 0

    for p in range( procNum ):

        recvCount[ p ] = binNum[ p ]

        recvOffset[ p ] = offsetSum

        offsetSum += binNum[ p ]

    return np.array( recvCount ), np.array( recvOffset )
