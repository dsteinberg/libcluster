#include <boost/math/special_functions.hpp>
#include "distributions.h"


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
// Private Functions
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


bool paircomp (const pair<int,double>& i, const pair<int,double> j)
{
  return i.second > j.second;
}


//
// Stick-Breaking (Dirichlet Process) parameter distribution.
//

distributions::StickBreak::StickBreak ()
  : alpha1_p(distributions::ALPHA1PRIOR),
    alpha2_p(distributions::ALPHA2PRIOR),
    Nk(ArrayXd::Zero(1)),
    alpha1(ArrayXd::Zero(1)),
    alpha2(ArrayXd::Zero(1)),
    E_logv(ArrayXd::Zero(1)),
    E_lognv(ArrayXd::Zero(1)),
    ordvec(1, pair<int,double>(0,0))
{

  // Prior free energy contribution
  this->F_p = lgamma(this->alpha1_p) + lgamma(this->alpha2_p)
              - lgamma(this->alpha1_p + this->alpha2_p);
}


void distributions::StickBreak::update (const ArrayXd& Nk)
{
  int K = Nk.size();

  // Destructively resize members to be the same size as Nk, no-op if same
  this->alpha1.resize(K);
  this->alpha2.resize(K);
  this->E_logv.resize(K);
  this->E_lognv.resize(K);
  this->E_logpi.resize(K);
  this->ordvec.resize(K, pair<int,double>(-1, -1));

  // Order independent update
  this->Nk     = Nk;
  this->alpha1 = this->alpha1_p + Nk;

  // Get at sort size order of clusters
  for (int k = 0; k < K; ++k)
  {
    this->ordvec[k].first  = k;
    this->ordvec[k].second = Nk(k);
  }
  sort(this->ordvec.begin(), this->ordvec.end(), paircomp);

  // Now do order dependent updates
  double N = Nk.sum(), cumNk = 0, cumE_lognv = 0, psisum;
  for (int idx = 0, k; idx < K; ++idx)
  {
    k = this->ordvec[idx].first;

    // Alpha 2
    cumNk += Nk(k);    // Accumulate cluster size sum
    this->alpha2(k) = this->alpha2_p + (N - cumNk);

    // Expected stick lengths
    psisum = digamma(this->alpha1(k) + this->alpha2(k));
    this->E_logv(k)  = digamma(this->alpha1(k)) - psisum;
    this->E_lognv(k) = digamma(this->alpha2(k)) - psisum;

    // Expected weights
    this->E_logpi(k) = this->E_logv(k) + cumE_lognv;
    cumE_lognv += E_lognv(k);         // Accumulate log stick length left
  }
}


double distributions::StickBreak::fenergy () const
{
  int K = this->alpha1.size();

  return K * this->F_p + (mxlgamma(this->alpha1 + this->alpha2).array()
          - mxlgamma(this->alpha1).array() - mxlgamma(this->alpha2).array()
          + (this->alpha1 - this->alpha1_p) * this->E_logv
          + (this->alpha2 - this->alpha2_p) * this->E_lognv).sum();
}


//
// Generalised Dirichlet parameter distribution.
//

void distributions::GDirichlet::update (const ArrayXd& Nk)
{
  // Call base class (stick breaking) update
  this->StickBreak::update(Nk);
  int smallk = (this->ordvec.end() - 1)->first; // Get smallest & last cluster

  // Set last stick lengths to 1 ( log(0) = 1 ) and adjust log marginal
  this->E_logpi(smallk) = this->E_logpi(smallk) - this->E_logv(smallk);
  this->E_logv(smallk)  = 0; // exp(E[log v_K]) = 1
  this->E_lognv(smallk) = 0; // Undefined, but set to zero
}


double distributions::GDirichlet::fenergy () const
{
  int K = this->ordvec.size();

  // GDir only has K-1 parameters, so we don't calculate the last F contrib.
  double Fpi = 0;
  for (int idx = 0, k = 0; idx < K-1; ++idx)
  {
    k = this->ordvec[idx].first;
    Fpi += lgamma(this->alpha1(k) + this->alpha2(k))
           - lgamma(this->alpha1(k)) - lgamma(this->alpha2(k))
           + (this->alpha1(k) - this->alpha1_p) * this->E_logv(k)
           + (this->alpha2(k) - this->alpha2_p) * this->E_lognv(k);
  }

  return (K-1) * this->F_p + Fpi;
}


//
// Dirichlet parameter distribution.
//

distributions::Dirichlet::Dirichlet ()
  : alpha_p(distributions::ALPHA1PRIOR),
    Nk(ArrayXd::Zero(1)),
    alpha(ArrayXd::Zero(1))
{}


void distributions::Dirichlet::update (const ArrayXd& Nk)
{
  int K = Nk.size();

  // Destructively resize members to be the same size as Nk, no-op if same
  this->alpha.resize(K);
  this->E_logpi.resize(K);

  // Hyperparameter update
  this->Nk    = Nk;
  this->alpha = this->alpha_p + Nk;

  // Expectation update
  this->E_logpi = mxdigamma(this->alpha).array() - digamma(this->alpha.sum());
}


double distributions::Dirichlet::fenergy () const
{
  int K = this->alpha.size();

  return lgamma(this->alpha.sum()) - (this->alpha_p-1) * this->E_logpi.sum()
      + ((this->alpha-1) * this->E_logpi - mxlgamma(this->alpha).array()).sum()
      - lgamma(K * this->alpha_p) + K * lgamma(this->alpha_p);
}


//
// Gaussian Wishart parameter distribution.
//

distributions::GaussWish::GaussWish (
    const double clustwidth,
    const RowVectorXd& meanX,
    const MatrixXd& covX
    )
  : beta_p(distributions::BETAPRIOR),
    nu(0),
    beta(0),
    logdW(0),
    N(0)
{
  int D = meanX.cols();

  // Create Prior
  this->nu_p = D;
  this->m_p  = meanX;

  try
  {
    VectorXd eigvec;
    double eigval = eigpower(covX, eigvec);
    this->iW_p    = eigval * this->nu_p * clustwidth * MatrixXd::Identity(D, D);
    this->logdW_p = -logdet(this->iW_p);
  }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }

  // Calculate prior free energy contribution
  this->F_p = mxlgamma((this->nu_p + 1
              - enumdims(this->m_p.cols())).matrix() / 2).sum();

  // Initialise Posterior to zeros, and other variables
  this->m  = RowVectorXd::Zero(D);
  this->iW = MatrixXd::Zero(D, D);
}


distributions::GaussWish::GaussWish (
    const double cwidthp,
    const RowVectorXd& cmeanp
    )
  : beta_p(distributions::BETAPRIOR),
    nu(0),
    beta(0),
    logdW(0),
    N(0)
{
  int D = cmeanp.cols();

  if (cwidthp <= 0)
    throw invalid_argument("cwidthp must be > 0!");

  this->nu_p = D;
  this->m_p  = cmeanp;
  this->iW_p = this->nu_p * cwidthp *  MatrixXd::Identity(D, D);
  try
    { this->logdW_p = -logdet(this->iW_p); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }

  // Calculate prior free energy contribution
  this->F_p = mxlgamma((this->nu_p + 1
              - enumdims(this->m_p.cols())).matrix() / 2).sum();

  // Initialise Posterior to zeros, and other variables
  this->m  = RowVectorXd::Zero(D);
  this->iW = MatrixXd::Zero(D, D);
}


void distributions::GaussWish::update (
      double N,
      const RowVectorXd& x_s,
      const MatrixXd& xx_s
    )
{
  // Prepare the Sufficient statistics
  RowVectorXd xk = RowVectorXd::Zero(x_s.cols());
  if (N > 0)
    xk = x_s/N;
  MatrixXd Sk = xx_s - xk.transpose()*x_s;
  RowVectorXd xk_m = xk - this->m_p;               // for iW, (xk - m)

  // Update posterior params
  this->N    = N;
  this->nu   = this->nu_p + N;
  this->beta = this->beta_p + N;
  this->m    = (this->beta_p * this->m_p + x_s) / this->beta;
  this->iW.noalias() = this->iW_p + Sk
               + (this->beta_p*N/this->beta) * xk_m.transpose() * xk_m;
  try
    { this->logdW = -logdet(this->iW); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Calc log(det(W)): ").append(e.what())); }
}


VectorXd distributions::GaussWish::Eloglike (const MatrixXd& X) const
{
  int D = X.cols();

  // Expectations of log Gaussian likelihood
  VectorXd E_logX(X.rows());
  double sumpsi = mxdigamma((this->nu + 1 - enumdims(D)).matrix() / 2).sum();
  try
  {
    E_logX.noalias() = 0.5 * (sumpsi + this->logdW - D*(1/this->beta + log(pi))
             - this->nu * mahaldist(X, this->m, this->iW).array()).matrix();
  }
  catch (invalid_argument e)
    { throw(string("Calculating Gaussian likelihood: ").append(e.what())); }

  return E_logX;
}


ArrayXb distributions::GaussWish::splitobs (const MatrixXd& X) const
{

  // Find the principle eigenvector using the power method if not done so
  VectorXd eigvec;
  eigpower(this->iW, eigvec);

  // 'split' the observations perpendicular to this eigenvector.
  return (((X.rowwise() - this->m)
             * eigvec.asDiagonal()).array().rowwise().sum()) >= 0;
}


double distributions::GaussWish::fenergy () const
{
  int D = this->m.cols();
  ArrayXd l = enumdims(D);
  double sumpsi = mxdigamma((this->nu + 1 - l).matrix() / 2).sum();

  return this->F_p + 0.5 * (D * (this->beta_p/this->beta - 1 - this->nu
          - log(this->beta_p/this->beta)) + this->nu
          * ((this->iW.ldlt().solve(this->iW_p)).trace()+this->beta_p
          * mahaldist(this->m, this->m_p, this->iW).coeff(0,0))
          + this->nu_p * (this->logdW_p - this->logdW)
          + this->N*sumpsi) - mxlgamma((this->nu+1-l).matrix() / 2).sum();
}


void distributions::GaussWish::getmeancov (
    RowVectorXd& mean,
    MatrixXd& cov
    ) const
{
  mean = this->m;
  cov  = this->iW/this->nu;
}
