#include "libcluster.h"
#include "probutils.h"
#include "testdata.h"

//
// Namespaces
//


using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;


//
// Functions
//


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd X1, X2, Xcat;
  vector<MatrixXd> X;
  makedata(X1, X2, Xcat, X);


  // GMC
  GMM gmm;
  double F;
  vector<RowVectorXd> w;
  vector<MatrixXd> qZgroup;
  clock_t start = clock();
  F = learnGMC (X, qZgroup, w, gmm, true, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "GMC Elapsed time = " << stop << " sec." << endl;

  for (unsigned int j = 0; j < X.size(); ++j)
    cout << "w" << j << " = " << w[j] << endl;
  cout << endl << gmm << endl;


  // SGMC
  start = clock();
  F = learnSGMC (X, qZgroup, w, gmm, true, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Symmetric GMC Elapsed time = " << stop << " sec." << endl;

  for (unsigned int j = 0; j < X.size(); ++j)
    cout << "w" << j << " = " << w[j] << endl;
  cout << endl << gmm << endl;


  // VDP
  MatrixXd qZ;
  start = clock();
  learnVDP(Xcat, qZ, gmm, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "VDP Elapsed time = " << stop << " sec." << endl;
  cout << endl << gmm << endl;


  // GMM
  start = clock();
  learnGMM(Xcat, qZ, gmm, true);

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Bayesian GMM Elapsed time = " << stop << " sec." << endl;
  cout << endl << gmm << endl;

  return 0;
}
