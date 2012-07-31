#include "testdata.h"
#include "test.h"

//
// Namespaces
//


using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vMatrixXd X;
  makedata(Xcat, X);

  const int J = X.size();

  // GMC
  SuffStat SS;
  vSuffStat SSgroup(J, SuffStat());
  vMatrixXd qZgroup;
  clock_t start = clock();
  learnGMC (X, qZgroup, SSgroup, SS, true, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "GMC Elapsed time = " << stop << " sec." << endl;
  printwj(SSgroup);
  printGMM(SS);


  // SGMC
  SS = SuffStat();
  SSgroup.clear();
  start = clock();
  learnSGMC (X, qZgroup, SSgroup, SS, true, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Symmetric GMC Elapsed time = " << stop << " sec." << endl;
  printwj(SSgroup);
  printGMM(SS);


  // VDP
  MatrixXd qZ;
  SS = SuffStat();
  start = clock();
  learnVDP(Xcat, qZ, SS, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "VDP Elapsed time = " << stop << " sec." << endl;
  printGMM(SS);


  // GMM
  start = clock();
  SS = SuffStat();
  learnBGMM(Xcat, qZ, SS, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Bayesian GMM Elapsed time = " << stop << " sec." << endl;
  printGMM(SS);

  return 0;
}
