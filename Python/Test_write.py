import numpy as np
import functions

filename0 = "test0.txt"
avgFilename0 = "test.avg0.txt"
filename1 = "test1.txt"
avgFilename1 = "test.avg1.txt"
filename2 = "test2.txt"
avgFilename2 = "test.avg2.txt"

data0 = np.arange( 24 ).reshape( 4, 6 )
avg0 = np.average( data0, axis=0 )
err0 = np.std( data0, axis=0 )

data1 = np.arange( 24 ).reshape( 6, 4 )
avg1 = np.average( data1, axis=1 )
err1 = np.std( data1, axis=0 )

data2 = np.arange( 24 ).reshape( 2, 3, 4 )
avg2 = np.average( data2, axis=0 )
err2 = np.std( data2, axis=0 )

print "0"

functions.writeDataFile( data0, filename0 )
functions.writeAvgDataFile( avg0, err0, avgFilename0 )

print "1"

functions.writeDataFile( data1, filename1 )
functions.writeAvgDataFile( avg1, err1, avgFilename1 )

print "2"

functions.writeDataFile( data2, filename2 )
functions.writeAvgDataFile( avg2, err2, avgFilename2 )
