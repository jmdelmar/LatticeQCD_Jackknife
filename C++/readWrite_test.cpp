#include "readWrite.h"

using namespace std;

void readTwop_0mom(double *data, char *file, char *dataset) {

  herr_t status;

  hsize_t dim[3];
  dim[0] = 32;
  dim[1] = 2109;
  dim[2] = 2;

  int subDimInfo[3];
  subDimInfo[0] = 32;
  subDimInfo[1] = 1;
  subDimInfo[2] = 2;

  hsize_t subDim = subDimInfo;

  float buff[32][1][2];

  hsize_t count[3];
  hsize_t offset[3];
  hsize_t stride[3];
  hsize_t block[3];

  count[0] = 32;
  count[1] = 1;
  count[2] = 2;

  offset[0] = 0;
  offset[1] = 0;
  offset[2] = 0;

  stride[0] = 1;
  stride[1] = 1;
  stride[2] = 1;

  block[0] = 1;
  block[1] = 1;
  block[2] = 1;

  hid_t file_id = H5Fopen (file, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen2 (file_id, dataset, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (3, subDim, NULL); 
  hid_t dataspace_id = H5Dget_space (dataset_id);

  status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset, stride, count, block);  

  status = H5Dread (dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, buff);

  status = H5Sclose (memspace_id);
  status = H5Sclose (dataspace_id);
  status = H5Dclose (dataset_id);
  status = H5Fclose (file_id);

  for(int i=0; i<32; i++)
    for(int j=0; j<2; j++) {

      data[2*i + j] = buff[i][0][j];

    }

  return;
}
