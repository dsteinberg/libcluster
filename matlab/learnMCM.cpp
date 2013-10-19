/*
 * libcluster -- A collection of Bayesian clustering algorithms
 * Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

/*! \brief Matlab interface to the Multiple-source Clustering Model (MCM) model
 *       clustering algorithm.
 *
 * \param nlhs number of outputs.
 * \param plhs outputs:
 *          - plhs[0], qY, {Jx[IjxT]} cell array of top-level cluster labels
 *          - plhs[1], qZ, {Jx{Ijx[NijxK]}} nested cells of bottom-level cluster 
 *              labels
 *          - plhs[2], weights_j, {Jx[1xT]} Group top-level cluster weights
 *          - plhs[3], weights_t, [TxK] Top-level cluster proportsion parameters
 *          - plhs[4], top-level cluster means, {Kx[1xD]} (Gaussian)
 *          - plhs[5], top-level cluster covariances, {Kx[DxD]} (Gaussian)
 *          - plhs[6], bottom-level cluster means, {Kx[1xD]} (Gaussian)
 *          - plhs[7], bottom-level cluster covariances, {Kx[DxD]} (Gaussian)
 * \param nrhs number of input arguments.
 * \param prhs input arguments:
 *          - prhs[0], W, {Jx[IjxD1]} nested cells of observation matrices
 *          - prhs[1], X, {Jx{Ijx[NijxD2]}} nested cells of observation matrices
 *          - prhs[2], options structure, with members:
 *              + trunc, [unsigned int] truncation level for top-level clusters
 *              + prior, [double] prior value corresponsing to W
 *              + prior2, [double] prior value corresponding to X
 *              + verbose, [bool] verbose output flag
 *              + sparse, [bool] do fast but approximate sparse VB updates
 *              + threads, [unsigned int] number of threads to use
 */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Parse some inputs
  if (nrhs < 2)                     // Enough input arguments?
    mexErrMsgTxt("Need at least some input data, W and X.");

  // Map W and X matlab matrices to constant eigen matrices
  const vMatrixXd W = cell2vec(prhs[0]);
  const vvMatrixXd X = cellcell2vecvec(prhs[1]);

  // Create and parse the options structure
  Options opts;
  if (nrhs > 2)
    opts.parseopts(prhs[2]);

  // redirect cout
  mexstreambuf mexout;
  mexout.hijack();

  // Run the algorithm
  vector<GDirichlet> weights_j;
  vector<Dirichlet>  weights_t;
  vector<GaussWish>  clusters_t;
  vector<GaussWish>  clusters_k;
  vMatrixXd qY;
  vvMatrixXd qZ;

  try
  {
    learnMCM(W, X, qY, qZ, weights_j, weights_t, clusters_t, clusters_k,
             opts.trunc, opts.prior, opts.prior2, opts.verbose, opts.threads);
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
  plhs[4] = mxCreateCellMatrix(1, T);
  plhs[5] = mxCreateCellMatrix(1, T);

  for (unsigned int t = 0; t < T; ++t)
  {
    mxSetCell(plhs[3], t, eig2mat(weights_t[t].Elogweight().exp()));
    mxSetCell(plhs[4], t, eig2mat(clusters_t[t].getmean()));
    mxSetCell(plhs[5], t, eig2mat(clusters_t[t].getcov()));
  }

  // Bottom-level Cluster Parameters
  const unsigned int K = clusters_k.size();
  plhs[6] = mxCreateCellMatrix(1, K);
  plhs[7] = mxCreateCellMatrix(1, K);

  for (unsigned int k = 0; k < K; ++k)
  {
    mxSetCell(plhs[6], k, eig2mat(clusters_k[k].getmean()));
    mxSetCell(plhs[7], k, eig2mat(clusters_k[k].getcov()));
  }
}
