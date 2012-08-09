#include "libcluster.h"
#include "distributions.h"
#include "testdata.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace distributions;


//
// Functions
//


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vMatrixXd X;
  vvMatrixXd Xv(2);
  makedata(Xcat, X);

  // Dived up X into 2 meta datasets, in an alternating fashion
  for (unsigned int j = 0; j < X.size(); ++j)
  {
    if ((j % 2) == 0)
      Xv[0].push_back(X[j]);
    else
      Xv[1].push_back(X[j]);
  }

  vector<GDirichlet> weights;
  vector<Dirichlet>  classes;
  vector<GaussWish>  clusters;
  vMatrixXd qY;
  vvMatrixXd qZ;
  clock_t start = clock();

  learnTCM(Xv, qY, qZ, weights, classes, clusters, 4, PRIORVAL, true);

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
