import numpy as np
import lqcdjk_fitting as fit

data = np.array( range(12) ).reshape(3,4)

err = np.random.rand( 3 )

fit, chiSq = fit.fitPlateau( data, err, 1, 2 )

print( fit )
