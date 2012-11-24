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

/*! \brief Matlab interface to the Variational Dirichlet Process (VDP)
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
  StickBreak weights;
  vector<GaussWish> clusters;
  MatrixXd qZ;

  try
  {
    learnVDP(X, qZ, weights, clusters, opts.prior, opts.verbose, opts.threads);
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
