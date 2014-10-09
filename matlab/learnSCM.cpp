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

#include <stdexcept>
#include "mex.h"
#include "libcluster.h"
#include "distributions.h"
#include "mintfctns.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace distributions;


//
// Interface
//

/*! \brief Matlab interface to the Simultaneous Clustering Model (TCM) model
 *       clustering algorithm.
 *
 * \param nlhs number of outputs.
 * \param plhs outputs:
 *          - plhs[0], qY, {Jx[IjxT]} cell array of top-level cluster labels
 *          - plhs[1], qZ, {Jx{Ijx[NijxK]}} nested cells of bottom-level cluster 
 *                     labels
 *          - plhs[2], weights, {Jx[1xT]} Group top-level cluster weights
 *          - plhs[3], weights_t, [TxK] Top-level cluster parameters
 *                     (bottom-level cluster weights_t)
 *          - plhs[4], means, {Kx[1xD]} Gaussian cluster means
 *          - plhs[5], covariances, {Kx[DxD]} Gaussian cluster covariances
 * \param nrhs number of input arguments.
 * \param prhs input arguments:
 *          - prhs[0], X, {Jx{Ijx[NijxD]}} nested cells of observation matrices
 *          - prhs[1], options structure, with members:
 *              + trunc, [unsigned int] truncation level for top-level clusters
 *              + prior, [double] prior value corresponsing to top-level clust.
 *              + prior2, [double] prior value corresponding to X (bottom-level
 *                  clusters)
 *              + verbose, [bool] verbose output flag
 *              + sparse, [bool] do fast but approximate sparse VB updates
 *              + threads, [unsigned int] number of threads to use
 */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Parse some inputs
  if (nrhs < 1)                     // Enough input arguments?
    mexErrMsgTxt("Need at least some input data, X.");

  // Map X matlab matrices to constant eigen matrices
  const vvMatrixXd X = cellcell2vecvec(prhs[0]);

  // Create and parse the options structure
  Options opts;
  if (nrhs > 1)
    opts.parseopts(prhs[1]);

  // redirect cout
  mexstreambuf mexout;
  mexout.hijack();

  // Run the algorithm
  vector<GDirichlet> weights_j;
  vector<Dirichlet> weights_t;
  vector<GaussWish> clusters;
  vMatrixXd qY;
  vvMatrixXd qZ;

  try
  {
    learnSCM(X, qY, qZ, weights_j, weights_t, clusters, opts.trunc, opts.prior,
             opts.prior2, opts.verbose, opts.threads);
  }
  catch (exception& e)
  {
    mexErrMsgTxt(e.what());
    mexout.restore();
  }

  // Restore cout
  mexout.restore();

  // Now format the returns - Most of this is memory copying. This is because
  //  safety has been chosen over more complex, but memory efficient methods.

  // Assignments
  plhs[0] = vec2cell(qY);
  plhs[1] = vecvec2cellcell(qZ);

  // Weights
  const unsigned int J = weights_j.size();
  plhs[2] = mxCreateCellMatrix(1, J);

  for (unsigned int j = 0; j < J; ++j)
    mxSetCell(plhs[2], j, eig2mat(weights_j[j].Elogweight().exp()));

  // Top-level Cluster Parameters
  const unsigned int T = weights_t.size();
  plhs[3] = mxCreateCellMatrix(1, T);

  for (unsigned int t = 0; t < T; ++t)
    mxSetCell(plhs[3], t, eig2mat(weights_t[t].Elogweight().exp()));

  // Bottom-level Cluster Parameters
  const unsigned int K = clusters.size();
  plhs[4] = mxCreateCellMatrix(1, K);
  plhs[5] = mxCreateCellMatrix(1, K);

  for (unsigned int k = 0; k < K; ++k)
  {
    mxSetCell(plhs[4], k, eig2mat(clusters[k].getmean()));
    mxSetCell(plhs[5], k, eig2mat(clusters[k].getcov()));
  }
}


