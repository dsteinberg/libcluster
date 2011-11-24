#ifndef VBCOMMON_H
#define VBCOMMON_H

#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include "libcluster.h"
#include "distributions.h"


/*! Functions and that implement commonly occuring routines. */
namespace vbcommon
{


//
// Namespace 'symbolic' constants
//

const int    SPLITITER   = 20;          //!< Max number of iter. for split VBEM
const double CONVERGE    = 1.0e-5;      //!< Convergence threshold
const double FENGYDEL    = CONVERGE/10; //!< Allowance for +ve F.E. steps
const double ZEROCUTOFF  = 0.1;         //!< Obs. count cut off sparse updates


/*! \brief Partition the observations, X into only those with
 *         p(z_n = k|x_n) > 0.5.
 *
 *  \param X NxD matrix of observations.
 *  \param qZk Nx1 vector of probabilities p(z_n = k|x_n).
 *  \param Xk a MxD matrix of observations that belong to the cluster with
 *         p > 0.5 (mutable).
 *  \param map a Mx1 array of the locations of Xk in X (mutable).
 */
void partX (
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& qZk,
    Eigen::MatrixXd& Xk,
    Eigen::ArrayXi& map
    );


/*! \brief Augment the assignment matrix, qZ with the split cluster entry.
 *         The new cluster assignemnts are put in the K+1 th column in the
 *         return matrix
 *
 *  \param k the cluster to split (i.e. which column of qZ)
 *  \param map an index mapping from the array of split assignments to the
 *         observations in qZ. map[idx] -> n.
 *  \param split the boolean array of split assignments.
 *  \param qZ the [NxK] observation assignment probability matrix.
 *  \returns The new observation assignments, [Nx(K+1)].
 *  \throws std::invalid_argument if map.size() != split.size().
 */
Eigen::MatrixXd augmentqZ (
    const double k,
    const Eigen::ArrayXi& map,
    const ArrayXb& split,
    const Eigen::MatrixXd& qZ
    );


/*! Check if a vector of ClustDist parameter distributions have any dists
 *    without observations contributing to their posteriors.
 *
 * \param cdists is a vector of posterior distribution classes. These classes
 *    must have a getN() member function that returns a double (a count of
 *    observations).
 *  \returns True if any of the posterior distributions have observations
 *           associated with them, false otherwise.
 */
template<class C> bool anyempty (const std::vector<C>& cdists)
{
  for (unsigned int k = 0; k < cdists.size(); ++k)
    if (cdists[k].getN() <= 1)
      return true;

  return false;
}


/*! Make a Gaussian mixture model object (GMM) from the Gaussian-Wishart
 *    distributions.
 *
 * \param cdists is the vector of posterior Gaussian-Wishart distributions.
 * \returns the GMM object.
 */
libcluster::GMM makeGMM (const std::vector<distributions::GaussWish>& cdists);

}

#endif // VBCOMMON_H
