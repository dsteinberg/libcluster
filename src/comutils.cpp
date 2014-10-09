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

#include "comutils.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;
using namespace distributions;


//
// Public Functions
//

void comutils::arrfind (
    const ArrayXb& expression,
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


ArrayXi comutils::partobs (
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


MatrixXd  comutils::auglabels (
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
