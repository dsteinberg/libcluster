#include "libcluster.h"
#include "probutils.h"
#include "testdata.h"
#include "test.h"

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
  MatrixXd Xcat;
  vMatrixXd X;
  makedata(Xcat, X);

  const unsigned I = X.size();

  SuffStat SS;
  vSuffStat SSdocs(I, SuffStat());
  MatrixXd qY, classparams;
  vMatrixXd qZ;
  clock_t start = clock();

  learnTCM(X, qY, qZ, SSdocs, SS, classparams, 4, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Topic Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Class proportions:" << endl;
  cout << qY.colwise().sum()/I << endl << endl;

  cout << endl << "Class parameters:" << endl;
  cout << classparams << endl << endl;

  cout << "Document cluster proportions:";
  printwj(SSdocs);
  cout << endl << "Cluster parameters:";
  printGMM(SS);

//  cout << endl << qY << endl;

  return 0;
}
