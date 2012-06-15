#include "comutils.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;


//
// Public Functions
//

void comutils::arrfind (
    const probutils::ArrayXb& expression,
    ArrayXi& indtrue,
    ArrayXi& indfalse
    )
{
  const int N = expression.size(),
            M = expression.count();

  indtrue.setZero(M);
  indfalse.setZero(N-M);

  for (int n = 0, m = 0, l = 0; n < N; ++n)
    expression(n) ? indtrue(m++) = n : indfalse(l++) = n;
}


ArrayXi comutils::partX (
    const MatrixXd& X,
    const ArrayXb& Xpart,
    MatrixXd& Xk
    )
{
  const int M = Xpart.count();

  ArrayXi pidx, npidx;
  comutils::arrfind(Xpart, pidx, npidx);

  Xk.setZero(M, X.cols());
  for (int m=0; m < M; ++m)           // index copy X to Xk
    Xk.row(m) = X.row(pidx(m));

  return pidx;
}


MatrixXd  comutils::augmentqZ (
    const double k,
    const ArrayXi& map,
    const ArrayXb& Zsplit,
    const MatrixXd& qZ
    )
{
  const int K = qZ.cols(),
            S = Zsplit.count();

  if (Zsplit.size() != map.size())
    throw invalid_argument("map and split must be the same size!");

  // Create new qZ for all data with split
  MatrixXd qZaug = qZ;    // Copy the existing qZ into the new
  qZaug.conservativeResize(Eigen::NoChange, K+1);
  qZaug.col(K).setZero();

  ArrayXi sidx, nsidx;
  comutils::arrfind(Zsplit, sidx, nsidx);

  // Copy split cluster assignments (augment qZ effectively)
  for (int s = 0; s < S; ++s)
  {
    qZaug(map(sidx(s)), K) = qZ(map(sidx(s)), k); // Add new cluster onto end
    qZaug(map(sidx(s)), k) = 0;
  }

  return qZaug;
}


bool comutils::anyempty (const SuffStat& SS)
{
  const int K = SS.getK();

  for (int k = 0; k < K; ++k)
    if (SS.getN_k(k) <= 1)
      return true;

  return false;
}
