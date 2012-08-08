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
    if (SS.getNk(k) <= 1)
      return true;

  return false;
}


/*  Find and remove all empty clusters. This is now necessary if we don't do an
 *    exhaustive search for the BEST cluster to split.
 *
 *    returns: true if any clusters have been deleted, false if all are kept.
 *    mutable: qZ may have columns deleted if there are empty clusters found.
 *    mutable: SSgroups if there are empty clusters found.
 *    mutable: SS if there are empty clusters found.
 */
bool comutils::prune_clusters (
    vMatrixXd& qZ,        // Probabilities qZ
    vSuffStat& SSgroups,  // Sufficient stats of groups
    SuffStat& SS          // Sufficient stats
    )
{
  const unsigned int K = SS.getK(),
                     J = qZ.size();

  // Look for empty sufficient statistics
  ArrayXd Nk = ArrayXd::Zero(K);
  for (unsigned int k = 0; k < K; ++k)
    Nk(k) = SS.getNk(k);

  // Find location of empty and full clusters
  ArrayXi eidx, fidx;
  arrfind(Nk.array() < ZEROCUTOFF, eidx, fidx);
  const unsigned int nempty = eidx.size();

  // If everything is not empty, return false
  if (nempty == 0)
    return false;

  // Delete empty cluster suff. stats.
  for (unsigned int i = nempty - 1; i >= 0; --i)
  {
    SS.delk(eidx(i));
    for (unsigned int j = 0; j < J; ++j)
      SSgroups[j].delk(eidx(i));
  }

  // Delete empty cluster indicators by copying only full indicators
  const unsigned int newK = fidx.size();
  vMatrixXd newqZ(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    newqZ[j].setZero(qZ[j].rows(), newK);
    for (unsigned int k = 0; k < newK; ++k)
      newqZ[j].col(k) = qZ[j].col(fidx(k));
  }

  qZ = newqZ;

  return true;
}
