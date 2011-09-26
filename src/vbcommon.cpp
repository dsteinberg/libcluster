// TODO get rid of friend classes, make interface functions

#include "vbcommon.h"
#include "probutils.h"
#include <boost/math/special_functions.hpp>


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace boost::math;


//
//  File scope variables
//

// Define pi
const double pi = constants::pi<double>(); // Boost high precision pi


//
// Private Prototypes
//

ArrayXd enumdims (int D); // Used to enumerate dimensions, returns {1,2,...,D}


//
// SBprior class method definitions
//

vbcommon::SBprior::SBprior ()
{
  // Stick breaking priors (Beta)
  this->alpha1 = ALPHA1PRIOR;
  this->alpha2 = ALPHA2PRIOR;

  // prior group stick breaking log expectations
  double psisum = digamma(this->alpha1 + this->alpha2);
  this->E_logv  = digamma(this->alpha1) - psisum;
  this->E_lognv = digamma(this->alpha2) - psisum;
}


double vbcommon::SBprior::fnrgprior () const
{
  return lgamma(this->alpha1) + lgamma(this->alpha2)
          - lgamma(this->alpha1 + this->alpha2);
}


//
// SBposterior class method definitions
//

void vbcommon::SBposterior::update (
    const double Nk,
    const double Ninf,
    const SBprior& prior
  )
{
  this->alpha1 = prior.getalpha1() + Nk;
  this->alpha2 = prior.getalpha2() + Ninf;
}


double vbcommon::SBposterior::fnrgpost (const SBprior& prior) const
{
  return lgamma(this->alpha1 + this->alpha2)
          - lgamma(this->alpha1) - lgamma(this->alpha2)
          + (this->alpha1 - prior.getalpha1()) * this->E_logv
          + (this->alpha2 - prior.getalpha2()) * this->E_lognv;
}


double vbcommon::SBposterior::Eloglike (double cumE_lognv, bool truncate)
{
  if (truncate == false)
  {
    double psisum = digamma(this->alpha1 + this->alpha2);
    this->E_logv  = digamma(this->alpha1) - psisum;
    this->E_lognv = digamma(this->alpha2) - psisum;
  }
  else
  {
    this->E_logv  = 0; // exp(E[log v_K]) = 1
    this->E_lognv = 0; // Undefined, but we set to zero so does not affect F.
  }

  // Expectations of log mixture weights (order sensitive)
  this->E_logZ = this->E_logv + cumE_lognv;

  return this->E_lognv;
}


//
// GWprior class method definitions
//

vbcommon::GWprior::GWprior (
    const double cwidthp,
    const Eigen::RowVectorXd& cmeanp
    )
{
  if (cwidthp <= 0)
    throw invalid_argument("cwidthp must be > 0!");

  int D      = cmeanp.cols();
  this->beta = BETAPRIOR;
  this->nu   = D;
  this->m    = cmeanp;
  this->iW   = this->nu * cwidthp *  MatrixXd::Identity(D, D);
  try
    { this->logdW = -logdet(this->iW); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }
}


vbcommon::GWprior::GWprior (
    const double clustwidth,
    const MatrixXd& covX,
    const RowVectorXd& meanX
    )
{
  int D = meanX.cols();

  this->beta = BETAPRIOR;
  this->nu   = D;
  this->m    = meanX;

  try
  {
    VectorXd eigvec;
    double eigval = eigpower(covX, eigvec);
    this->iW      = eigval * this->nu * clustwidth * MatrixXd::Identity(D, D);
    this->logdW   = -logdet(this->iW);
  }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }
}


double vbcommon::GWprior::fnrgprior () const
{
  return mxlgamma((this->nu+1-enumdims(this->m.cols())).matrix() / 2).sum();
}


//
// GWposterior class method definitions
//

void vbcommon::GWposterior::update (
    const double Nk,
    const RowVectorXd& xksum,
    const MatrixXd& Rksum,
    const GWprior& prior
    )
{
  // Prepare the Sufficient statistics
  RowVectorXd xk = RowVectorXd::Zero(xksum.cols());
  if (Nk > 0)
    xk = xksum/Nk;
  MatrixXd Sk = Rksum - xk.transpose()*xksum;
  RowVectorXd xk_m = xk - prior.getrefm();               // for iW, (xk - m)

  // Update posterior params
  this->Nk   = Nk;
  this->nu   = prior.getnu() + Nk;
  this->beta = prior.getbeta() + Nk;
  this->m    = (prior.getbeta() * prior.getrefm() + xksum) / this->beta;
  this->iW   = prior.getrefiW() + Sk
               + (prior.getbeta()*Nk/this->beta) * xk_m.transpose() * xk_m;
  try
    { this->logdW = -logdet(this->iW); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Calc log(det(W)): ").append(e.what())); }
}


double vbcommon::GWposterior::fnrgpost (const vbcommon::GWprior& prior) const
{
  int D = this->m.cols();
  ArrayXd l = enumdims(D);
  double sumpsi = mxdigamma((this->nu + 1 - l).matrix() / 2).sum();

  return 0.5 * (D * (prior.getbeta()/this->beta - 1 - this->nu
          - log(prior.getbeta()/this->beta)) + this->nu
          * ((this->iW.ldlt().solve(prior.getrefiW())).trace()+prior.getbeta()
          * mahaldist(this->m, prior.getrefm(), this->iW).coeff(0,0))
          + prior.getnu() * (prior.getlogdW() - this->logdW)
          + this->Nk*sumpsi) - mxlgamma((this->nu+1-l).matrix() / 2).sum();
}


VectorXd vbcommon::GWposterior::Eloglike (const MatrixXd& X) const
{
  int D = X.cols(),
      N = X.rows();

  // Expectations of log Gaussian likelihood
  VectorXd E_logX(N);
  double sumpsi = mxdigamma((this->nu + 1 - enumdims(D)).matrix() / 2).sum();
  try
  {
    E_logX = 0.5 * (sumpsi + this->logdW - D*(1/this->beta + log(pi))
             - this->nu * mahaldist(X, this->m, this->iW).array()).matrix();
  }
  catch (invalid_argument e)
    { throw(string("Calculating Gaussian likelihood: ").append(e.what())); }

  return E_logX;
}


//
// Public Function Definitions
//

void vbcommon::partX (
    const MatrixXd& X,
    const VectorXd& qZk,
    MatrixXd& Xk,
    ArrayXi& map
    )
{
  int D = X.cols(),
      N = X.rows();

  // Make a copy of the observations with only relevant data points, p > 0.5
  ArrayXb zidx = qZk.array() > 0.5;
  int M = zidx.count();

  map = ArrayXi::Zero(M);
  Xk  = MatrixXd::Zero(M,D);
  for (int n=0, m=0; n < N; ++n) // index copy X to Xk
  {
    if (zidx(n) == true)
    {
      Xk.row(m) = X.row(n);
      map(m) = n;
      ++m;
    }
  }
}


MatrixXd  vbcommon::augmentqZ (
    const double k,
    const ArrayXi& map,
    const ArrayXb& split,
    const MatrixXd& qZ
    )
{
  int K = qZ.cols(),
      N = qZ.rows(),
      M = map.rows();

  if (split.size() != M)
    throw invalid_argument("map and split must be the same size!");

  // Create new qZ for all data with split
  MatrixXd qZk = MatrixXd::Zero(N, K+1);

  // Copy the existing qZ into the new
  qZk.leftCols(K) = qZ;

  // Copy split cluster assignments (augment qZ effectively)
  for (int m=0; m < M; ++m)
  {
    if (split(m) == true)
    {
      qZk(map(m), K) = qZ(map(m), k); // Add new cluster onto the end
      qZk(map(m), k) = 0;
    }
  }

  return qZk;
}


bool vbcommon::paircomp (const pair<int,double>& i, const pair<int,double> j)
{
  return i.second > j.second;
}


//
// Private Function Definitions
//

ArrayXd enumdims (int D)
{
  ArrayXd l;

  if (D > 1)
    l = ArrayXd::LinSpaced(D, 1, D);
  else
    l = ArrayXd::Ones(1);

  return l;
}
