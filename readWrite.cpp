#include "readWrite.h"

using namespace lQCD_jk;
using std::cout;
using std::endl;

void readTwopMesons_0mom(vector< vector<double> > *data, char *file, char *dataset, readInfo info) {

  herr_t status;

  int dimNum = info.dimNum;

  // if( info.subDim.size() != dimNum ) {

  hsize_t subDim[dimNum];

  for(int d=0; d<dimNum; d++)
    subDim[d] = info.subDim[d];

  float buff[ subDim[0] ][ subDim[1] ][ subDim[2] ];

  hsize_t count[dimNum];
  hsize_t offset[dimNum];
  hsize_t stride[dimNum];
  hsize_t block[dimNum];

  for(int d=0; d<dimNum; d++)
    count[d] = info.count[d];

  for(int d=0; d<dimNum; d++)
    offset[d] = info.offset[d];

  for(int d=0; d<dimNum; d++)
    stride[d] = info.stride[d];

  for(int d=0; d<dimNum; d++)
    block[d] = info.block[d];

  hid_t file_id = H5Fopen (file, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen2 (file_id, dataset, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (dimNum, subDim, NULL); 
  hid_t dataspace_id = H5Dget_space (dataset_id);

  status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset, stride, count, block);  

  status = H5Dread (dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, buff);

  status = H5Sclose (memspace_id);
  status = H5Sclose (dataspace_id);
  status = H5Dclose (dataset_id);
  status = H5Fclose (file_id);

  for(int i=0; i<subDim[0]; i++)
    for(int j=0; j<subDim[2]; j++)
      data->at(i).push_back( buff[i][0][j] );

  return;
}

void readTwopMesons_0mom(vector< vector<double> > *data, char *file, char *dataset, readInfo info) {

  herr_t status;

  int dimNum = info.dimNum;

  // if( info.subDim.size() != dimNum ) {

  hsize_t subDim[dimNum];

  for(int d=0; d<dimNum; d++)
    subDim[d] = info.subDim[d];

  float buff[ subDim[0] ][ subDim[1] ][ subDim[2] ];

  hsize_t count[dimNum];
  hsize_t offset[dimNum];
  hsize_t stride[dimNum];
  hsize_t block[dimNum];

  for(int d=0; d<dimNum; d++)
    count[d] = info.count[d];

  for(int d=0; d<dimNum; d++)
    offset[d] = info.offset[d];

  for(int d=0; d<dimNum; d++)
    stride[d] = info.stride[d];

  for(int d=0; d<dimNum; d++)
    block[d] = info.block[d];

  hid_t file_id = H5Fopen (file, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen2 (file_id, dataset, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (dimNum, subDim, NULL); 
  hid_t dataspace_id = H5Dget_space (dataset_id);

  status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset, stride, count, block);  

  status = H5Dread (dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, buff);

  status = H5Sclose (memspace_id);
  status = H5Sclose (dataspace_id);
  status = H5Dclose (dataset_id);
  status = H5Fclose (file_id);

  for(int i=0; i<subDim[0]; i++)
    for(int j=0; j<subDim[2]; j++)
      data->at(i).push_back( buff[i][0][j] );

  return;
}
