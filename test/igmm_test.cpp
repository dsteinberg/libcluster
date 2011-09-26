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

  // Make an I-GMC object
  IGMC igmc(2, 2, 0.2, 2, RowVectorXd::Zero(2));

  int i = 0;
  bool conv;

  clock_t start = clock();

  do
  {
    cout << "Iteration: " << ++i << endl;

    conv = false;

    conv = learnIGMC(X1, igmc, true);
    conv = learnIGMC(X2, igmc, true);

    cout << "tau = " << igmc.gettau() << endl << endl;
  }
  while (conv == true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << endl << "GMC Elapsed time = " << stop << " sec." << endl;

  MatrixXd qZ1;
  RowVectorXd w1 = classifyIGMC(X1, igmc, qZ1, true);
  cout << "w1 = " << w1 << endl << endl;

  MatrixXd qZ2;
  RowVectorXd w2 = classifyIGMC(X2, igmc, qZ2, true);
  cout << "w2 = " << w2 << endl << endl;

  cout << igmc << endl;
}
