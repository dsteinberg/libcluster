#include <boost/math/special_functions.hpp>
#include "distributions.h"
#include "probutils.h"

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
// Private Helper Functions
//

/* Compare an <int,double> double pair by the double member. Useful
 *  for sorting an array in descending order while retaining a notion of
 *  the original order of the array.
 *
 *  returns: true if i.second > j.second.
 */
bool inline paircomp (
    const std::pair<int,double>& i, // the first pair to compare.
    const std::pair<int,double>& j  // the second pair to compare.
    )
{
  return i.second > j.second;
}


/* Enumerate the dimensions.
 *
 *  returns: 1:D or if D = 1, return 1.
 */
ArrayXd enumdims (int D)
{
  ArrayXd l;

  if (D > 1)
    l = ArrayXd::LinSpaced(D, 1, D);
  else
    l = ArrayXd::Ones(1);

  return l;
}


//
// Stick-Breaking (Dirichlet Process) weight distribution.
//

distributions::StickBreak::StickBreak ()
  : alpha1_p(distributions::ALPHA1PRIOR),
    alpha2_p(distributions::ALPHA2PRIOR),
    Nk(ArrayXd::Zero(1)),
    alpha1(ArrayXd::Ones(1)*distributions::ALPHA1PRIOR),
    alpha2(ArrayXd::Ones(1)*distributions::ALPHA2PRIOR),
    E_logv(ArrayXd::Zero(1)),
    E_lognv(ArrayXd::Zero(1)),
    E_logpi(ArrayXd::Zero(1)),
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
// Generalised Dirichlet weight distribution.
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
// Dirichlet weight distribution.
//

distributions::Dirichlet::Dirichlet ()
  : alpha_p(distributions::ALPHA1PRIOR),
    Nk(ArrayXd::Zero(1)),
    alpha(ArrayXd::Ones(1)*distributions::ALPHA1PRIOR),
    E_logpi(ArrayXd::Zero(1))
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
// Gaussian Wishart cluster distribution.
//

distributions::GaussWish::GaussWish (
    const double clustwidth,
    const unsigned int D
    )
  : ClusterDist(clustwidth, D),
    nu_p(D),
    beta_p(distributions::BETAPRIOR),
    m_p(RowVectorXd::Zero(D)),
    nu(D),
    beta(distributions::BETAPRIOR),
    m(RowVectorXd::Zero(D)),
    logdW(0),
    N(0)
{
  if (clustwidth <= 0)
    throw invalid_argument("clustwidth must be > 0!");

  // Create Prior
  this->iW_p = this->iW = this->nu_p * this->prior * MatrixXd::Identity(D, D);

  try
    { this->logdW_p = this->logdW = -logdet(this->iW_p); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }

  // Calculate prior free energy contribution
  this->F_p = mxlgamma((this->nu_p + 1
              - enumdims(this->m_p.cols())).matrix() / 2).sum();
}


void distributions::GaussWish::makeSS (
    const VectorXd& qZk,
    const MatrixXd& X,
    MatrixXd& x_s,
    MatrixXd& xx_s
    )
{
  MatrixXd qZkX = qZk.asDiagonal() * X;
  x_s  = qZkX.colwise().sum();  // [1xD] row vector
  xx_s = qZkX.transpose() * X;  // [DxD] matrix
}


Array4i distributions::GaussWish::dimSS (const Eigen::MatrixXd& X)
{
  return Array4i(1, X.cols(), X.cols(), X.cols());
}

void distributions::GaussWish::update (
      double N,
      const MatrixXd& x_s,
      const MatrixXd& xx_s
    )
{
  if (
      (x_s.rows() != 1)
      || (x_s.cols() != this->D)
      || (xx_s.cols() != this->D)
      || (xx_s.rows() != this->D)
     )
    throw invalid_argument("Suff. Stats. are wrong dim. for updating params!");

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
  // Expectations of log Gaussian likelihood
  VectorXd E_logX(X.rows());
  double sumpsi = mxdigamma((this->nu+1-enumdims(this->D)).matrix()/2).sum();
  try
  {
    E_logX.noalias() = 0.5 * (sumpsi + this->logdW - this->D*(1/this->beta
      + log(pi)) - this->nu * mahaldist(X, this->m, this->iW).array()).matrix();
  }
  catch (invalid_argument e)
    { throw(string("Calculating Gaussian likelihood: ").append(e.what())); }

  return E_logX;
}


distributions::ArrayXb distributions::GaussWish::splitobs (
    const MatrixXd& X
    ) const
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
  ArrayXd l = enumdims(this->D);
  double sumpsi = mxdigamma((this->nu + 1 - l).matrix() / 2).sum();

  return this->F_p + (this->D * (this->beta_p/this->beta - 1 - this->nu
          - log(this->beta_p/this->beta))
          + this->nu * ((this->iW.ldlt().solve(this->iW_p)).trace()
          + this->beta_p * mahaldist(this->m, this->m_p, this->iW).coeff(0,0))
          + this->nu_p * (this->logdW_p - this->logdW) + this->N*sumpsi)/2
          - mxlgamma((this->nu+1-l).matrix() / 2).sum();
}


//
// Normal Gamma parameter distribution.
//

distributions::NormGamma::NormGamma (
    const double clustwidth,
    const unsigned int D
    )
  : ClusterDist(clustwidth, D),
    nu_p(distributions::NUPRIOR),
    beta_p(distributions::BETAPRIOR),
    m_p(RowVectorXd::Zero(D)),
    nu(distributions::NUPRIOR),
    beta(distributions::BETAPRIOR),
    m(RowVectorXd::Zero(D)),
    N(0)
{
  if (clustwidth <= 0)
    throw invalid_argument("clustwidth must be > 0!");

  // Create Prior
  this->L_p = this->L = this->nu_p * this->prior * RowVectorXd::Ones(D);
  this->logL_p = this->L_p.array().log().sum();
}


void distributions::NormGamma::makeSS (
    const VectorXd& qZk,
    const MatrixXd& X,
    MatrixXd& x_s,
    MatrixXd& xx_s
    )
{
  MatrixXd qZkX = qZk.asDiagonal() * X;
  x_s  = qZkX.colwise().sum();                        // [1xD] row vector
  xx_s = (qZkX.array() * X.array()).colwise().sum();  // [1xD] row vector
}


Array4i distributions::NormGamma::dimSS (const Eigen::MatrixXd& X)
{
  return Array4i(1, X.cols(), 1, X.cols());
}


void distributions::NormGamma::update (
      double N,
      const MatrixXd& x_s,
      const MatrixXd& xx_s
    )
{
  if (
      (x_s.rows() != 1)
      || (x_s.cols() != this->D)
      || (xx_s.cols() != this->D)
      || (xx_s.rows() != 1)
     )
    throw invalid_argument("Suff. Stats. are wrong dim. for updating params!");

  // Prepare the Sufficient statistics
  RowVectorXd xk = RowVectorXd::Zero(x_s.cols());
  if (N > 0)
    xk = x_s/N;
  RowVectorXd Sk = xx_s.array() - x_s.array().square()/N;

  // Update posterior params
  this->N    = N;
  this->beta = this->beta_p + N;
  this->nu   = this->nu_p + N/2;
  this->m    = (this->beta_p * this->m_p + x_s) / this->beta;
  this->L    = this->L_p + Sk/2 + (this->beta_p * N / (2 * this->beta))
                * (xk - this->m_p).array().square().matrix();
  this->logL = this->L.array().log().sum();
}


VectorXd distributions::NormGamma::Eloglike (const MatrixXd& X) const
{
  // Distance evaluation in the exponent
  VectorXd Xmdist = (X.rowwise() - this->m).array().square().matrix()
                      * this->L.array().inverse().matrix().transpose();

  // Expectations of log Gaussian likelihood
  return 0.5 * (this->D * (digamma(this->nu) - log(2 * pi) - 1/this->beta)
              - this->logL - this->nu * Xmdist.array());
}


distributions::ArrayXb distributions::NormGamma::splitobs (
    const MatrixXd& X
    ) const
{
  // Find location of largest element in L, this is the 'eigenvector'
  int eigvec;
  this->L.maxCoeff(&eigvec);

  // 'split' the observations perpendicular to this 'eigenvector'.
  return (X.col(eigvec).array() - this->m(eigvec)) >= 0;
}


double distributions::NormGamma::fenergy () const
{
  VectorXd iL = this->L.array().inverse().matrix().transpose();

  return D*(lgamma(this->nu_p) - lgamma(this->nu)
    + this->N*digamma(this->nu)/2 - this->nu)
    + D/2 * (log(this->beta) - log(this->beta_p) - 1 + this->beta_p/this->beta)
    + this->beta_p*this->nu/2*(this->m - this->m_p).array().square().matrix()*iL
    + this->nu_p*(this->logL - this->logL_p) + this->nu*this->L_p*iL;
}
