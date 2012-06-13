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

// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vector<MatrixXd> X;
  makedata(Xcat, X);

  MatrixXd qY;
  vector<MatrixXd> qZ;
  clock_t start = clock();

  learnTOP(X, qY, qZ, 6, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Topic Elapsed time = " << stop << " sec." << endl;

  return 0;
}
