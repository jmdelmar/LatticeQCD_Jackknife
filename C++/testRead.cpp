#include <iostream>
#include "readWrite.h"

using namespace lQCD_jk;
using std::cout;
using std::endl;

int main (void) {

  char *file = "/home/tuf47161/work/L16T32/twop/1505-MG/twop.1505_mesons_Qsq64_SS.00.08.09.04.h5";
  char *dataset = "/conf_1505/sx00sy08sz09st04/g1/twop_meson_1";

  readInfo rInfo;

  info.timestepNum = 48;
  info.complex = 0; // get real part

  vector <vector< vector<double> > > data(rInfo.timestepNum);

  vector<string> confs(2);

  confs.at(0) = "0200";

  confs.at(1) = "0220";

  vector<

  readTwopMesons_0mom(&data, file, dataset, rInfo);

  for(int i=0; i<data.size(); i++)
    cout << data.at(i).at(0) << ", " << data.at(i).at(1) << endl;

  return 0;
}
