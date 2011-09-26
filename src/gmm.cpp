#include <boost/math/special_functions.hpp>
#include "libcluster.h"
#include "probutils.h"


//
// Namespaces
//


using namespace std;
using namespace Eigen;
using namespace boost::math;
using namespace probutils;


//
// Private Globals
//


// Define pi
const double pi = constants::pi<double>(); // Boost high precision pi


//
// Public Member Functions
//

libcluster::GMM::GMM () : D(0), K(0) {}


libcluster::GMM::GMM (
    const vector<RowVectorXd>& mu,
    const vector<MatrixXd>& sigma,
    const vector<double>& w
    )
{
  // Check for valid arguments
  if ((mu.size() != sigma.size()) || (mu.size() != sigma.size()))
    throw invalid_argument("Parameter vectors are not the same size!");
  else if (mu.size() == 0)
    throw invalid_argument("Need to have non-zero length parameter vectors.");

  // Assign values to properties
  this->mu = mu;
  this->sigma = sigma;
  this->w = w;
  this->K = w.size();
  this->D = mu[0].cols();
}


//
// Public Functions
//

MatrixXd libcluster::classify (const MatrixXd& X, const libcluster::GMM& gmm)
{
  int N = X.rows();
  int K = gmm.getK();
  int D = X.cols();

  // Check the data and GMM are compatible
  if (D != gmm.getD())
    throw invalid_argument("Data and GMM dimensionality not equal!");

  // Log likelihood of each weighted Gaussian mixture log( p(Z)p(X|Z) )
  MatrixXd loglike(N,K);
  for (int k=0; k < K; ++k)
  {
    try
    {
      loglike.col(k) = log(gmm.getw(k))
                       - 0.5 * ( D * log(2 * pi) + logdet(gmm.getsigma(k))
                       + mahaldist(X, gmm.getmu(k), gmm.getsigma(k)).array());
    }
    catch (invalid_argument e)
      { throw; }
  }

  // Make log probability log(p(Z|X)), i.e. normalise the log likelihoods
  MatrixXd logpZ = loglike.colwise() - logsumexp(loglike);

  return (logpZ.array().exp()).matrix(); // return pZ
}


VectorXd libcluster::predict (const MatrixXd& X, const libcluster::GMM& gmm)
{
  int N = X.rows();
  int K = gmm.getK();
  int D = X.cols();

  // Check the data and GMM are compatible
  if (D != gmm.getD())
    throw invalid_argument("Data and GMM dimensionality not equal!");

  // Log likelihood of each weighted Gaussian mixture log( p(Z)p(X|Z) )
  MatrixXd loglike(N,K);
  for (int k=0; k < K; ++k)
  {
    try
    {
      loglike.col(k) = log(gmm.getw(k))
                       - 0.5 * ( D * log(2 * pi) + logdet(gmm.getsigma(k))
                       + mahaldist(X, gmm.getmu(k), gmm.getsigma(k)).array());
    }
    catch (invalid_argument e)
      { throw; }
  }

  // Sum out p(Z) to make p(x*|X), use logsumexp to do this accurately
  return (logsumexp(loglike).array().exp()).matrix();
}


ostream& libcluster::operator<< (ostream& s, const libcluster::GMM& gmm)
{
  // K and D of the GMM
  s << "K = " << gmm.getK() << endl;
  s << "D = " << gmm.getD() << endl << endl;

  // Weights of each cluster
  s << "w = [ ";
  for (int k=0; k < gmm.getK(); ++k) { s << gmm.getw(k) << " "; }
  s << "]" << endl << endl;

  // Means of each cluster
  for (int k=0; k < gmm.getK(); ++k)
    s << "mu(" << (k+1) << ") = " << gmm.getmu(k) << endl;

  // Covariances of each cluster
  for (int k=0; k < gmm.getK(); ++k)
    s << "\nsigma(" << (k+1) << ") =\n" << gmm.getsigma(k) << endl;

  return s;
}
