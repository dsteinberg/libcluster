#include <stdexcept>
#include "mex.h"
#include "libcluster.h"
#include "distributions.h"
#include "intfctns.h"


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

/*! \brief Matlab interface to the Grouped Mixtures Clustering (GMC) model
 *         clustering algorithm.
 *
 * \param nlhs number of outputs.
 * \param plhs outputs:
 *          - plhs[0], qZ, {Jx[NxK]} cell array of assignment probablities
 *          - plhs[1], weights, {Jx[1xK]} Group mixture weights
 *          - plhs[2], means, {Kx[1xD]} Gaussian mixture means
 *          - plhs[3], covariances, {Kx[DxD]} Gaussian mixture covariances
 * \param nrhs number of input arguments.
 * \param prhs input arguments:
 *          - prhs[0], X, {Jx[NxD]} cell array of observation matrices
 *          - prhs[1], options structure, with members:
 *              + prior, [double] prior value
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
  const vMatrixXd X = cell2vec(prhs[0]);

  // Create and parse the options structure
  Options opts;
  if (nrhs > 1)
    opts.parseopts(prhs[1]);

  // redirect cout
  mexstreambuf mexout;
  mexout.hijack();

  // Run the algorithm
  vector<GDirichlet> weights;
  vector<GaussWish> clusters;
  vMatrixXd qZ;

  try
  {
    learnGMC(X, qZ, weights, clusters, opts.prior, opts.sparse, opts.verbose,
             opts.threads);
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

  // Assignments
  plhs[0] = vec2cell(qZ);

  // Weights
  const unsigned int J = weights.size();
  plhs[1] = mxCreateCellMatrix(1, J);

  for (unsigned int j = 0; j < J; ++j)
    mxSetCell(plhs[1], j, eig2mat(weights[j].Elogweight().exp()));

  // Cluster Parameters
  const unsigned int K = clusters.size();
  plhs[2] = mxCreateCellMatrix(1, K);
  plhs[3] = mxCreateCellMatrix(1, K);

  for (unsigned int k = 0; k < K; ++k)
  {
    mxSetCell(plhs[2], k, eig2mat(clusters[k].getmean()));
    mxSetCell(plhs[3], k, eig2mat(clusters[k].getcov()));
  }
}

