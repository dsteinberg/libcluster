
#include <limits>
#include "libcluster.h"
#include "probutils.h"
#include "distributions.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace distributions;


//
//  Private Functions
//

/* Batch Variational Bayes EM for all group mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument rethrown from other functions or if cdists.size()
 *          does not match qZ[j].cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class L, class C> double vbem (
    const vector<MatrixXd>& X,  // Observations Jx[NjxD]
    vector<MatrixXd>& qZ,       // Observations to cluster assignments Jx[NjxK]
    MatrixXd& qY,               // Indicator to label assignments [JxT]
    W wdists,                   // Model class weights
    vector<L> ldists,           // Class parameters
    vector<C> cdists,           // Cluster parameters
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false  // Verbose output (default false)
    )
{
  const int J = X.size(),
            K = qZ[0].cols(),
            T = qY.cols();

  // Construct the parameters
//  W wdists;
//  vector<L> ldists(T, L());
//  vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

  double F = numeric_limits<double>::max(), Fold;
  vector<double> Fxz(J);
  int i = 0;

  do
  {
    Fold = F;

    // VBM for class weights
    wdists.update(qY.colwise().sum());

    // VBM for class parameters
    MatrixXd Njk = MatrixXd::Zero(J,K);
    for (int j=0; j<J; ++j)
      Njk.row(j) = qZ[j].colwise().sum();   // Sum over groups to get multinoms.

    for (int t=0; t < T; ++t)
      ldists[t].update(qY.col(t).transpose()*Njk);  // Weighted multinomials.

    // VBM for cluster parameters
//    #pragma omp parallel for schedule(guided)
//    for (int k=0; k < K; ++k)
//      cdists[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));

    // VBE
//    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < J; ++j)
//      Fxz[j] = vbexpectation<W,C>(X[j], wdists[j], cdists, qZ[j], sparse);

    // Calculate free energy of model
//    F = fenergy<W,C>(wdists, cdists, Fxz, SSj, SS);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > libcluster::FENGYDEL)
      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > libcluster::CONVERGE)
          && ( (i++ < maxit) || (maxit < 0) ) );

  return F;
}
