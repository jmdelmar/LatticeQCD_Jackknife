import functions
import numpy as np

array1 = np.arange( 1, 25, dtype=float ).reshape( 4, 6 )

array2 = np.arange( 1, 25, dtype=float ). reshape( 2, 3, 4 )


print "array1:"

print array1

print functions.calculateEffMass( array1 )

print "array2:"

print array2

print functions.calculateEffMass( array2 )

