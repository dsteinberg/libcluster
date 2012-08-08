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
using namespace distributions;
using namespace probutils;


//
// Functions
//


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vvMatrixXd X(1);
  makedata(Xcat, X[0]);

  vector<GDirichlet> weights;
  vector<Dirichlet>  classes;
  vector<GaussWish>  clusters;
  vMatrixXd qY;
  vvMatrixXd qZ;
  clock_t start = clock();

  learnTCM(X, qY, qZ, weights, classes, clusters, 4, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Topic Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Class proportions:" << endl;
  for (vector<GDirichlet>::iterator j = weights.begin(); j < weights.end(); ++j)
    cout << j->Elogweight().exp().transpose() << endl;

  cout << endl << "Class parameters:" << endl;
  for (vector<Dirichlet>::iterator t = classes.begin(); t < classes.end(); ++t)
    cout << t->Elogweight().exp().transpose() << endl;

  cout << endl << "Cluster means:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getmean() << endl;

  cout << endl << "Cluster covariances:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getcov() << endl << endl;

  return 0;
}
