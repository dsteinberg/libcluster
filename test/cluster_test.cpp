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


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vMatrixXd X;
  makedata(Xcat, X);

  // GMC
  vector<GDirichlet> weights;
  vector<GaussWish>  clusters;
  vMatrixXd qZgroup;
  clock_t start = clock();
  learnGMC (X, qZgroup, weights, clusters, PRIORVAL, true, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "GMC Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Cluster Weights:" << endl;
  for (vector<GDirichlet>::iterator j = weights.begin(); j < weights.end(); ++j)
    cout << j->Elogweight().exp().transpose() << endl;

  cout << endl << "Cluster means:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getmean() << endl;

  cout << endl << "Cluster covariances:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getcov() << endl << endl;

  return 0;
}
