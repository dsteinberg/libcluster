#include "vbcommon.h"
#include "probutils.h"
#include <boost/math/special_functions.hpp>
#include <boost/math/constants/constants.hpp>


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace boost::math;
using namespace probutils;
using namespace distributions;


//
// Public Function Definitions
//

void vbcommon::partX (
    const MatrixXd& X,
    const VectorXd& qZk,
    MatrixXd& Xk,
    ArrayXi& map
    )
{
  int D = X.cols(),
      N = X.rows();

  // Make a copy of the observations with only relevant data points, p > 0.5
  ArrayXb zidx = qZk.array() > 0.5;
  int M = zidx.count();

  map = ArrayXi::Zero(M);
  Xk  = MatrixXd::Zero(M,D);
  for (int n=0, m=0; n < N; ++n) // index copy X to Xk
  {
    if (zidx(n) == true)
    {
      Xk.row(m) = X.row(n);
      map(m) = n;
      ++m;
    }
  }
}


MatrixXd  vbcommon::augmentqZ (
    const double k,
    const ArrayXi& map,
    const ArrayXb& split,
    const MatrixXd& qZ
    )
{
  int K = qZ.cols(),
      N = qZ.rows(),
      M = map.rows();

  if (split.size() != M)
    throw invalid_argument("map and split must be the same size!");

  // Create new qZ for all data with split
  MatrixXd qZk = MatrixXd::Zero(N, K+1);

  // Copy the existing qZ into the new
  qZk.leftCols(K) = qZ;

  // Copy split cluster assignments (augment qZ effectively)
  for (int m=0; m < M; ++m)
  {
    if (split(m) == true)
    {
      qZk(map(m), K) = qZ(map(m), k); // Add new cluster onto the end
      qZk(map(m), k) = 0;
    }
  }

  return qZk;
}


libcluster::GMM vbcommon::makeGMM (const vector<GaussWish>& cdists)
{
  int K = cdists.size();
  int N = 0;

  for (int k = 0; k < K; ++k)
    N += cdists[k].getN();

  vector<RowVectorXd> mu(K);
  vector<MatrixXd> sigma(K);
  vector<double> w(K);

  for (int k = 0; k < K; ++k)
  {
    cdists[k].getmeancov(mu[k], sigma[k]);
    w[k] = cdists[k].getN()/N;
  }

  libcluster::GMM retgmm(mu, sigma, w);
  return retgmm;
}
