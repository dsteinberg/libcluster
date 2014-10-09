/*
 * libcluster -- A collection of hierarchical Bayesian clustering algorithms.
 * Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
 *
 * This file is part of libcluster.
 *
 * libcluster is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * libcluster is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcluster. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef COMUTILS_H
#define COMUTILS_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "libcluster.h"
#include "probutils.h"
#include "distributions.h"


/*! Namespace that implements various common utilities used in the algorithms */
namespace comutils
{


//
// Helper structures
//

/* Triplet that contains the information for choosing a good cluster split
 *  ordering.
 */
struct GreedOrder
{
  int k;      // Cluster number/index
  int tally;  // Number of times a cluster has failed to split
  double Fk;  // The clusters approximate free energy contribution
};


//
// Helper functions
//

/* Compares two GreedOrder triplets and returns which is more optimal to split.
 *  Precendence is given to less split fail tally, and then to more free energy
 *  contribution.
 */
bool inline greedcomp (const GreedOrder& i, const GreedOrder& j)
{
  if (i.tally == j.tally)       // If the tally is the same, use the greater Fk
    return i.Fk > j.Fk;
  else if (i.tally < j.tally)   // Otherwise prefer the lower tally
    return true;
  else
    return false;
}


/* Find the indices of the ones and zeros in a binary array in the order they
 *  appear.
 *
 *  mutable: indtrue the indices of the true values in the array "expression"
 *  mutable: indfalse the indices of the false values in the array "expression"
 */
void arrfind (
    const distributions::ArrayXb& expression,
    Eigen::ArrayXi& indtrue,
    Eigen::ArrayXi& indfalse
    );


/* Partition the observations, X according to a logical array.
 *
 *  mutable: Xk, MxD matrix of observations that have a correspoding 1 in Xpart.
 *  returns: an Mx1 array of the locations of Xk in X.
 */
Eigen::ArrayXi partobs (
    const Eigen::MatrixXd& X,            // NxD matrix of observations.
    const distributions::ArrayXb& Xpart, // Nx1 indicator vector to partition X.
    Eigen::MatrixXd& Xk          // MxD matrix of obs. beloning to new partition
    );


/* Augment the assignment matrix, qZ with the split cluster entry.
 *
 * The new cluster assignments are put in the K+1 th column in the return matrix
 *  returns: The new observation assignments, [Nx(K+1)].
 *  throws: std::invalid_argument if map.size() != Zsplit.size().
 */
Eigen::MatrixXd  auglabels (
    const double k,               // Cluster to split (i.e. which column of qZ)
    const Eigen::ArrayXi& map,    // Mapping from array of partitioned obs to qZ
    const distributions::ArrayXb& Zsplit, // Boolean array of assignments.
    const Eigen::MatrixXd& qZ     // [NxK] observation assignment prob. matrix.
    );


/* Check if any sufficient statistics are empty.
 *
 *  returns: True if any of the sufficient statistics are empty
 */
template <class C> bool anyempty (const std::vector<C>& clusters)
{
  const unsigned int K = clusters.size();

  for (unsigned int k = 0; k < K; ++k)
    if (clusters[k].getN() <= 1)
      return true;

  return false;
}

}

#endif // COMUTILS_H
