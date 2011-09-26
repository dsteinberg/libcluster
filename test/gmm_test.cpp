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
  vector<MatrixXd> qZ;

  clock_t start = clock();

  try { F = learnGMC (X, qZ, w, gmm, true, true); }
  catch (runtime_error e) { throw e; }
  catch (logic_error e) { throw e; }

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "GMC Elapsed time = " << stop << " sec." << endl;

  for (unsigned int j = 0; j < X.size(); ++j)
  {
    cout << "w" << j << " = " << w[j] << endl;
  }
  cout << endl << gmm << endl;


  // VDP
  GMM gmm2;
  MatrixXd qZ2;
  start = clock();

  try { learnVDP(Xcat, qZ2, gmm2, true); }
  catch (runtime_error e) { throw e; }
  catch (logic_error e) { throw e; }

  stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "VDP Elapsed time = " << stop << " sec." << endl;

  cout << endl << gmm2 << endl;

  return 0;
}
