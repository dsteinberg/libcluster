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

/*! \brief Matlab interface to the Bayseian Gaussian Mixture Model (BGMM)
 *         clustering algorithm.
 *
 * \param nlhs number of outputs.
 * \param plhs outputs:
 *          - plhs[0], qZ, [NxK] assignment probablities
 *          - plhs[1], weights, [1xK] Gaussian mixture weights
 *          - plhs[2], means, {Kx[1xD]} Gaussian mixture means
 *          - plhs[3], covariances, {Kx[DxD]} Gaussian mixture covariances
 * \param nrhs number of input arguments.
 * \param prhs input arguments:
 *          - prhs[0], X, [NxD] observation matrix
 *          - prhs[1], options structure, with members:
 *              + prior, [double] prior value
 *              + verbose, [bool] verbose output flag
 *              + threads, [unsigned int] number of threads to use
 */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Parse some inputs
  if (nrhs < 1)                     // Enough input arguments?
    mexErrMsgTxt("Need at least some input data, X.");
  if (mxIsDouble(prhs[0]) == false) // Are the observations double precision?
    mexErrMsgTxt("X must be double precision!");

  // Map X matlab matrix to constant eigen matrix
  Map<const MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));

  // Create and parse the options structure
  Options opts;
  if (nrhs > 1)
    opts.parseopts(prhs[1]);

  // redirect cout
  mexstreambuf mexout;
  mexout.hijack();

  // Run the algorithm
  Dirichlet weights;
  vector<GaussWish> clusters;
  MatrixXd qZ;

  try
  {
    learnBGMM(X, qZ, weights, clusters, opts.prior, opts.verbose, opts.threads);
  }
  catch (exception e)
  {
    mexout.restore();
    mexErrMsgTxt(e.what());
  }

  // Restore cout
  mexout.restore();

  // Now format the returns - Most of this is memory copying. This is because
  //  safety has been chosen over more complex, but memory efficient methods.
  plhs[0] = eig2mat(qZ);                            // Assignments
  plhs[1] = eig2mat(weights.Elogweight().exp());    // Weights

  // Cluster Parameters
  unsigned int K = clusters.size();
  plhs[2] = mxCreateCellMatrix(1, K);
  plhs[3] = mxCreateCellMatrix(1, K);

  for (unsigned int k = 0; k < K; ++k)
  {
    mxSetCell(plhs[2], k, eig2mat(clusters[k].getmean()));
    mxSetCell(plhs[3], k, eig2mat(clusters[k].getcov()));
  }
}
