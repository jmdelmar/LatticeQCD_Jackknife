import h5py
import numpy as np
from os import listdir as ls
from glob import glob

def getSourcePositions( srcGroupName ):

    #srcPos_tmp = srcGroupName.split( "s" )[ 1: ]

    srcPos = srcGroupName.split( "s" )[ 1: ]

    dims = [ "x", "y", "z", "t" ]

    #srcPos = [ "", "", "", "" ]

    srcPos = [ src.replace( d, "" ) for src, d in zip( srcPos, dims ) ]

    #for src, tmp, d in zip( srcPos, srcPos_tmp, dims ):

    #src = tmp.replace( d, "" )

    return srcPos


def readFormFactorFile( filename, QsqNum, binNum, timestepNum ):

    with open( filename, "r" ) as file:

        data = np.array( file.read().split(), dtype=float )

    data = data.reshape( QsqNum, binNum, timestepNum, 3  )

    return data[ ..., -1 ]

# Reads HDF5 datsets containing the given keyword(s) if
# given and returns them as a numpy array. 

def getDatasets( configDir, configList, fn_template, *keyword ):
    
    data = []

    # Loop through config indexes

    for c in range( len( configList ) ):
        
        data.append( [] )

        # Get filenames in specific configs directory which follow template

        filename = glob( configDir + "/" + configList[c] + "/" + fn_template )

        filename = sorted( filename )

        # Loop through indexes of file names in specific config directory

        for fn in range( len( filename ) ): 

            data[c].append( [] )

            with h5py.File( filename[fn], "r" ) as dataFile:

                dsetname = []
        
                if keyword:

                    # Put all datasets which contain the first keyword into list
                    
                    dataFile.visititems( lambda name, obj: \
                                         dsetname.append( name ) \
                                         if type( obj ) is h5py.Dataset \
                                         and keyword[0] in name \
                                         else None )

                    # Fileter any datasets from list which do not contain 
                    # all of the keywords

                    for kw in keyword:

                        dsetname_filtered = []
                        
                        for ds in dsetname:

                            if kw in ds:

                                dsetname_filtered.append( ds )

                        dsetname = dsetname_filtered

                else:

                    # Put all datasets into list
                
                    dataFile.visititems( lambda name,obj: \
                                         dsetname.append(name) \
                                         if type( obj ) is h5py.Dataset \
                                         else None )

                topGroup_0 = dsetname[0].split( "/", 1 )

                data[ c ][ fn ].append( np.array( dataFile[ dsetname[ 0 ] ] ) )

                for name in dsetname[1:]:

                    topGroups = name.split( "/", 1 )

                    # Ensure that the top groups match across all files in sub-directory

                    assert ( topGroups[0] == topGroup_0[0] ), \
                        "Top groups in configuration " + configList[ c ] \
                        + " do not match" 
                    
                    data[ c ][ fn ].append( np.array( dataFile[ name ] ) )

                # End loop over datasets

            # Close file

        # End loop over files in sub-directory

    # End loop over configs

    return np.array( data )


def getHDF5File( configDir, configList, fn_template, *keyword ):

    dataset = []

    datasetName = []

    ins_current = [ "noether", "oneD", "ultra_local" ]

    for curr in ins_current:

        dset, name = getDatasets( configDir, configList, fn_template, curr, *keyword )

        dataset.append( dset )

        datasetName.append( name )

    return dataset, datasetName


def getHDF5File_wNames( configDir, configList, fn_template, *keyword ):

    dataset = []

    datasetName = []

    ins_current = [ "noether", "oneD", "ultra_local" ]

    for curr in ins_current:

        dset, name = getDatasets_wNames( configDir, configList, fn_template, curr, *keyword )

        dataset.append( dset )

        datasetName.append( name )

    return dataset, datasetName


# Reads HDF5 datsets containing the given keyword(s) if
# given and returns them as a numpy array. Also returns
# a list of the group/dataset names of each dataset

def getDatasets_wNames( configDir, configList, fn_template, *keyword ):

    dataset = []

    datasetName = []

    # Loop through config indexes

    for c in range( len( configList ) ):
        
        dataset.append( [] )
        
        datasetName.append( [] )
        
        # Get filenames in config sub-directory which follow template

        filename = glob( configDir + "/" + configList[c] + "/" + fn_template )

        filename = sorted( filename )

        fileNum = len( filename )

        sources = set()

        for fn in range( fileNum ): 

            dataset[ c ].append( [] )

            datasetName[ c ].append( [] )

            dsetName_tmp = []

            with h5py.File( filename[ fn ], "r" ) as dataFile:

                if keyword:

                    # Put all datasets which contain the first keyword into list
                    
                    dataFile.visititems( lambda name, obj: \
                                          dsetName_tmp.append( name ) \
                                          if type( obj ) is h5py.Dataset \
                                          and keyword[0] in name \
                                          else None )

                    # Filter any datasets from list which do not contain 
                    # all of the keywords

                    for kw in keyword:

                        dsetName_tmp_filtered = []
                        
                        for ds in dsetName_tmp:

                            if kw in ds:

                                dsetName_tmp_filtered.append( ds )

                        dsetName_tmp = dsetName_tmp_filtered

                else:

                    # Put all datasets into list
                
                    dataFile.visititems( lambda name,obj: \
                                          dsetName_tmp.append(name) \
                                          if type( obj ) is h5py.Dataset \
                                          else None )

                groups_0 = dsetName_tmp[0].split( "/" )

                sources.add( groups_0[1] )

                datasetName[ c ][ fn ].append( groups_0 )

                dataset[ c ][ fn ].append( np.array( dataFile[ dsetName_tmp[ 0 ] ] ) )
                
                for name in dsetName_tmp[1:]:

                    groups = name.split( "/" )

                    # Ensure that the top groups match across all files in sub-directory

                    assert ( groups[0] == groups_0[0] ), \
                        "Top groups in configuration " + configList[ c ] \
                        + " do not match" 
                    
                    sources.add( groups[1] )

                    datasetName[ c ][ fn ].append( groups )

                    dataset[ c ][ fn ].append( np.array( dataFile[ name ] ) )

                # End loop over datasets

            # Close file

        # End loop over files in sub-directory

        # Ensure that the source (2nd) group is unique to each file in sub-directory

        assert len( sources ) == fileNum, \
            "Source groups in configuration %s are not unique to file" % configList[ c ]

        # Ensure that the groups below the source group match across all files in sub-directory

        for fn in range( 1, fileNum ): 

            for ds in range( len ( datasetName[ c ][ fn ] ) ):

                assert datasetName[ c ][ 0 ][ ds ][ 2: ] == \
                             datasetName[ c ][ fn ][ ds ][ 2: ], \
                "Groups below source group in configuration " \
                + configList[ c ] + " do not match"

    # End loop over configs

    return np.array( dataset ), datasetName
    #return dataset, datasetName


def readDataFile( filename, timestepNum, binNum ):

    with open( filename, "r" ) as file:

        data = np.array( file.read().split(), dtype=float )

    data = data.reshape( binNum, timestepNum, 2  )

    return data[ ..., -1 ]

def readNthDataCol( filename, N ):

    data = []

    with open( filename, "r" ) as file:

        for line in file:

            data.append( line.split() )

    data = np.array( data, dtype=float )

    return data[ ..., N ]

def detTimestepAndConfigNum( filename ):

    t_last = -1

    timestepNum = 0

    timestepNum_last = -1

    configNum = 0

    with open( filename, "r" ) as file:

        for line in file:

            t = int( line.split()[ 0 ] )

            if t == ( t_last + 1 ):

                timestepNum += 1

                t_last = t

            elif t == 0:

                if timestepNum_last >= 0:

                    assert timestepNum == timestepNum_last, \
                        "Error (detTimestepAndConfigNum): Number of timesteps not" \
                        + " consistent across configurations"
                    
                timestepNum_last = timestepNum

                timestepNum = 1

                configNum += 1

                t_last = t

            else:

                print "Error (detTimestepAndConfigNum): Timestep in 1st column " \
                    + "does not behave as expected"

                return -1

    assert timestepNum == timestepNum_last, \
        "Error (detTimestepAndConfigNum): Number of timesteps not" \
        + " consistent across configurations"
                    
    configNum += 1

    return timestepNum, configNum


def detConfigAndTimestepNum( filename ):

    t_last = -1

    timestepNum = 0

    timestepNum_last = -1

    configNum = 0

    with open( filename, "r" ) as file:

        for line in file:

            t = int( line.split()[ 0 ] )

            if t == ( t_last + 1 ):

                timestepNum += 1

                t_last = t

            elif t == 0:

                if timestepNum_last >= 0:

                    assert timestepNum == timestepNum_last, \
                        "Error (detTimestepAndConfigNum): Number of timesteps not" \
                        + " consistent across configurations"
                    
                timestepNum_last = timestepNum

                timestepNum = 1

                configNum += 1

                t_last = t

            else:

                print "Error (detTimestepAndConfigNum): Timestep in 1st column " \
                    + "does not behave as expected"

                return -1

    assert timestepNum == timestepNum_last, \
        "Error (detTimestepAndConfigNum): Number of timesteps not" \
        + " consistent across configurations"
                    
    configNum += 1

    return configNum, timestepNum


def detQsqConfigNumAndTimestepNum( filename ):

    t_last = -1

    q_last = 0
    
    # We can set the 1st Qsq to zero because we will
    # check that it is equal to q_last

    Qsq = [ 0 ] 

    configNum = 0

    configNum_last = -1

    timestepNum = 0

    timestepNum_last = -1

    with open( filename, "r" ) as file:

        for line in file:

            t = int( line.split()[ 0 ] )

            q = int( line.split()[ 1 ] )

            #print str(t) + " " + str(q)

            if t == ( t_last + 1 ) and q == q_last:

                timestepNum += 1

            elif t == 0:

                if timestepNum_last >= 0:

                    assert timestepNum == timestepNum_last, \
                        "Error (detQsqConfigNumAndTimestepNum): Number of " \
                        + "timesteps is not consistent across configurations"
                    
                timestepNum_last = timestepNum

                timestepNum = 1

                configNum += 1

                if q > q_last:

                    if configNum_last >= 0:

                        assert configNum == configNum_last, \
                            "Error (detQsqConfigNumAndTimestepNum): Number of " \
                            + "configurations is not consistent across Qsq's"

                    configNum_last = configNum

                    configNum = 0

                    Qsq.append( q )

            else:

                print "Error (detTimestepAndConfigNum): Timestep in 1st column " \
                    + "does not behave as expected"

                return -1

            t_last = t

            q_last = q

    assert timestepNum == timestepNum_last, \
        "Error (detTimestepAndConfigNum): Number of timesteps not" \
        + " consistent across configurations"
                    
    configNum += 1

    assert configNum == configNum_last, \
        "Error (detQsqConfigNumAndTimestepNum): Number of \
        configurations is not consistent across Qsq's"

    return np.array( Qsq ), configNum, timestepNum


def writeDataFile( data, filename ):

    if data.ndim != 2:

        print "Error (writeDataFile): Data array does not have two dimensions"

        return -1

    with open( filename, "w" ) as output:

        for d0 in range( len( data ) ):

            for d1 in range( len( data[ d0 ] ) ):
                
                output.write( str( d1 ).ljust(5) + str( data[ d0, d1 ] ) + "\n" )


def writeAvgDataFile( data, error, filename ):

    if data.ndim != 1:

        print "Error (writeAvgDataFile): Data array has more than one dimension"

        return -1

    if data.shape != error.shape or len( data ) != len( error ):

        print "Error (writeAvgDataFile): Error array's length and shape does not match data array's"
        
        return -1

    with open( filename, "w" ) as output:

        for d0 in range( len( data ) ):

            output.write( str( d0 ).ljust(5) + str( data[ d0 ] ).ljust(20) + str( error[ d0 ] ) + "\n" )


def writeAvgDataFile_wX( x, y, error, filename ):

    if y.ndim != 1:

        print "Error (writeAvgDataFile): Data array has more than one dimension"

        return -1

    if y.shape != error.shape or len( y ) != len( error ):

        print "Error (writeAvgDataFile): Error array's length and shape does not match data array's"
        
        return -1

    with open( filename, "w" ) as output:

        for ix, iy, ierr in zip( x, y, error ):

            output.write( str( ix ).ljust(5) + str( iy ).ljust(20) + str( ierr ) + "\n" )


def writeFormFactorFile( filename, data, Qsq ):

    assert data.ndim == 3, "Error (writeFormFactorFile): " \
        + "Data array does not have three dimensions"

    with open( filename, "w" ) as output:

        for q in range( data.shape[ 0 ] ):

            for b in range( data.shape[ 1 ] ):
                
                for t in range( data.shape[ 2 ] ):

                    output.write( str( t ).ljust(5) + str( Qsq[ q ] ).ljust(5) + str( data[ q, b, t ] ) + "\n" )


def writeAvgFormFactorFile( filename, data, error, Qsq ):

    assert data.ndim == 2, "Error (writeAvgFormFactorFile): " \
        + "Data array does not have two dimensions"

    assert data.shape == error.shape, "Error (writeAvgFormFactorFile): " \
        + "Error array's shape does not match data array's"
        
    with open( filename, "w" ) as output:

        for q in range( data.shape[ 0 ] ):

            for t in range( data.shape[ 1 ] ):

                output.write( str( t ).ljust(5) 
                              + str( Qsq[ q ] ).ljust(5) 
                              + str( data[ q, t ] ).ljust(20) 
                              + str( error[ q, t ] ) + "\n" )


def writeFitDatafile( filename, fit, err, fitStart, fitEnd ):

    with open( filename, "w" ) as output:

        output.write( str( fit ).ljust( 20 ) 
                      + str( err ).ljust( 20 ) 
                      + str( int( fitStart ) ).ljust( 5 ) 
                      + str( int( fitEnd ) ) + "\n" )
