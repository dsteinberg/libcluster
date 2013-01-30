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
bool inline obscomp (
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
ArrayXd enumdims (const int D)
{
  ArrayXd l;

  if (D > 1)
    l = ArrayXd::LinSpaced(D, 1, D);
  else
    l.setOnes(1);

  return l;
}


//
// Stick-Breaking (Dirichlet Process) weight distribution.
//

distributions::StickBreak::StickBreak ()
  : WeightDist(),
    alpha1_p(distributions::ALPHA1PRIOR),
    alpha2_p(distributions::ALPHA2PRIOR),
    alpha1(ArrayXd::Constant(1, distributions::ALPHA1PRIOR)),
    alpha2(ArrayXd::Constant(1, distributions::ALPHA2PRIOR)),
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
  const int K = Nk.size();

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
  sort(this->ordvec.begin(), this->ordvec.end(), obscomp);

  // Now do order dependent updates
  const double N = Nk.sum();
  double cumNk = 0, cumE_lognv = 0;
  for (int idx = 0, k; idx < K; ++idx)
  {
    k = this->ordvec[idx].first;

    // Alpha 2
    cumNk += Nk(k);    // Accumulate cluster size sum
    this->alpha2(k) = this->alpha2_p + (N - cumNk);

    // Expected stick lengths
    double psisum    = digamma(this->alpha1(k) + this->alpha2(k));
    this->E_logv(k)  = digamma(this->alpha1(k)) - psisum;
    this->E_lognv(k) = digamma(this->alpha2(k)) - psisum;

    // Expected weights
    this->E_logpi(k) = this->E_logv(k) + cumE_lognv;
    cumE_lognv += E_lognv(k);         // Accumulate log stick length left
  }
}


double distributions::StickBreak::fenergy () const
{
  const int K = this->alpha1.size();

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
  const int smallk = (this->ordvec.end() - 1)->first; // Get smallest cluster

  // Set last stick lengths to 1 ( log(0) = 1 ) and adjust log marginal
  this->E_logpi(smallk) = this->E_logpi(smallk) - this->E_logv(smallk);
  this->E_logv(smallk)  = 0; // exp(E[log v_K]) = 1
  this->E_lognv(smallk) = 0; // Undefined, but set to zero
}


double distributions::GDirichlet::fenergy () const
{
  const int K = this->ordvec.size();

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
  : WeightDist(),
    alpha_p(distributions::ALPHA1PRIOR),
    alpha(ArrayXd::Constant(1, distributions::ALPHA1PRIOR)),
    E_logpi(ArrayXd::Zero(1))
{}


void distributions::Dirichlet::update (const ArrayXd& Nk)
{
  const int K = Nk.size();

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
  const int K = this->alpha.size();

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
    m_p(RowVectorXd::Zero(D))
{
  if (clustwidth <= 0)
    throw invalid_argument("clustwidth must be > 0!");

  // Create Prior
  this->iW_p = this->nu_p * this->prior * MatrixXd::Identity(D, D);

  try
    { this->logdW_p = -logdet(this->iW_p); }
  catch (invalid_argument e)
    { throw invalid_argument(string("Creating prior: ").append(e.what())); }

  // Calculate prior free energy contribution
  this->F_p = mxlgamma((this->nu_p + 1
              - enumdims(this->m_p.cols())).matrix() / 2).sum();

  this->clearobs(); // Empty suff. stats. and set posteriors equal to priors
}


void distributions::GaussWish::addobs(const VectorXd& qZk, const MatrixXd& X)
{
  if (X.cols() != this->D)
    throw invalid_argument("Mismatched dims. of cluster params and obs.!");
  if (qZk.rows() != X.rows())
    throw invalid_argument("qZk and X ar not the same length!");

  MatrixXd qZkX = qZk.asDiagonal() * X;

  this->N_s += qZk.sum();
  this->x_s += qZkX.colwise().sum();             // [1xD] row vector
  this->xx_s.noalias() += qZkX.transpose() * X;  // [DxD] matrix
}


void distributions::GaussWish::update ()
{
  // Prepare the Sufficient statistics
  RowVectorXd xk = RowVectorXd::Zero(this->D);
  if (this->N_s > 0)
    xk = this->x_s/this->N_s;
  MatrixXd Sk = this->xx_s - xk.transpose() * this->x_s;
  RowVectorXd xk_m = xk - this->m_p;               // for iW, (xk - m)

  // Update posterior params
  this->N    = this->N_s;
  this->nu   = this->nu_p + this->N;
  this->beta = this->beta_p + this->N;
  this->m    = (this->beta_p * this->m_p + this->x_s) / this->beta;
  this->iW   = this->iW_p + Sk
                + (this->beta_p * this->N/this->beta) * xk_m.transpose() * xk_m;

  try
    { this->logdW = -logdet(this->iW); }
  catch (invalid_argument e)
    { throw runtime_error(string("Calc log(det(W)): ").append(e.what())); }
}


void distributions::GaussWish::clearobs ()
{
  // Reset parameters back to prior values
  this->nu    = this->nu_p;
  this->beta  = this->beta_p;
  this->m     = this->m_p;
  this->iW    = this->iW_p;
  this->logdW = this->logdW_p;

  // Empty sufficient statistics
  this->N_s  = 0;
  this->x_s  = RowVectorXd::Zero(D);
  this->xx_s = MatrixXd::Zero(D,D);
}


VectorXd distributions::GaussWish::Eloglike (const MatrixXd& X) const
{
  // Expectations of log Gaussian likelihood
  VectorXd E_logX(X.rows());
  double sumpsi = mxdigamma((this->nu+1-enumdims(this->D)).matrix()/2).sum();
  try
  {
    E_logX = 0.5 * (sumpsi + this->logdW - this->D * (1/this->beta + log(pi))
                 - this->nu * mahaldist(X, this->m, this->iW).array()).matrix();
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
  const ArrayXd l = enumdims(this->D);
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
    m_p(RowVectorXd::Zero(D))
{
  if (clustwidth <= 0)
    throw invalid_argument("clustwidth must be > 0!");

  // Create Prior
  this->L_p = this->nu_p * this->prior * RowVectorXd::Ones(D);
  this->logL_p = this->L_p.array().log().sum();

  this->clearobs(); // Empty suff. stats. and set posteriors equal to priors
}


void distributions::NormGamma::addobs (const VectorXd& qZk, const MatrixXd& X)
{
  if (X.cols() != this->D)
    throw invalid_argument("Mismatched dims. of cluster params and obs.!");
  if (qZk.rows() != X.rows())
    throw invalid_argument("qZk and X ar not the same length!");

  MatrixXd qZkX = qZk.asDiagonal() * X;

  this->N_s  += qZk.sum();
  this->x_s  += qZkX.colwise().sum();                                 // [1xD]
  this->xx_s += (qZkX.array() * X.array()).colwise().sum().matrix();  // [1xD]
}


void distributions::NormGamma::update ()
{
  // Prepare the Sufficient statistics
  RowVectorXd xk = RowVectorXd::Zero(this->D);
  RowVectorXd Sk = RowVectorXd::Zero(this->D);
  if (this->N_s > 0)
  {
    xk = this->x_s/this->N_s;
    Sk = this->xx_s.array() - this->x_s.array().square()/this->N_s;
  }

  // Update posterior params
  this->N    = this->N_s;
  this->beta = this->beta_p + this->N;
  this->nu   = this->nu_p + this->N/2;
  this->m    = (this->beta_p * this->m_p + x_s) / this->beta;
  this->L    = this->L_p + Sk/2 + (this->beta_p * this->N / (2 * this->beta))
                * (xk - this->m_p).array().square().matrix();

  if ((this->L.array() <= 0).any())
    throw invalid_argument(string("Calc log(L): Variance is zero or less!"));

  this->logL = this->L.array().log().sum();
}


void distributions::NormGamma::clearobs ()
{
  // Reset parameters back to prior values
  this->nu   = this->nu_p;
  this->beta = this->beta_p;
  this->m    = this->m_p;
  this->L    = this->L_p;
  this->logL = this->logL_p;

  // Empty sufficient statistics
  this->N_s  = 0;
  this->x_s  = RowVectorXd::Zero(this->D);
  this->xx_s = RowVectorXd::Zero(this->D);
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
  const VectorXd iL = this->L.array().inverse().matrix().transpose();

  return D*(lgamma(this->nu_p) - lgamma(this->nu)
    + this->N*digamma(this->nu)/2 - this->nu)
    + D/2 * (log(this->beta) - log(this->beta_p) - 1 + this->beta_p/this->beta)
    + this->beta_p*this->nu/2*(this->m - this->m_p).array().square().matrix()*iL
    + this->nu_p*(this->logL - this->logL_p) + this->nu*this->L_p*iL;
}


//
// Exponential Gamma parameter distribution.
//

distributions::ExpGamma::ExpGamma (const double obsmag, const unsigned int D)
  : ClusterDist(obsmag, D),
    a_p(distributions::APRIOR),
    b_p(obsmag)
{
    this->clearobs(); // Empty suff. stats. and set posteriors equal to priors
}


void distributions::ExpGamma::addobs (const VectorXd& qZk, const MatrixXd& X)
{
  if (X.cols() != this->D)
    throw invalid_argument("Mismatched dims. of cluster params and obs.!");
  if (qZk.rows() != X.rows())
    throw invalid_argument("qZk and X ar not the same length!");

  this->N_s += qZk.sum();
  this->x_s += (qZk.asDiagonal() * X).colwise().sum();
}


void distributions::ExpGamma::update ()
{
  // Update posterior params
  this->N    = this->N_s;
  this->a    = this->a_p + this->N;
  this->ib   = (this->b_p + this->x_s.array()).array().inverse().matrix();
  this->logb = - this->ib.array().log().sum();
}


void distributions::ExpGamma::clearobs ()
{
  // Reset parameters back to prior values
  this->a    = this->a_p;
  this->ib   = RowVectorXd::Constant(this->D, 1/this->b_p);
  this->logb = this->D * log(this->b_p);

  // Empty sufficient statistics
  this->N_s = 0;
  this->x_s = RowVectorXd::Zero(this->D);
}


VectorXd distributions::ExpGamma::Eloglike (const MatrixXd& X) const
{
  return this->D * digamma(this->a) - this->logb
          - (this->a * X * this->ib.transpose()).array();
}


distributions::ArrayXb distributions::ExpGamma::splitobs (
    const MatrixXd& X
    ) const
{
  ArrayXd XdotL = X * (this->a * this->ib).transpose();
  return (XdotL > (XdotL.sum()/XdotL.size()));
}


double distributions::ExpGamma::fenergy () const
{
 return this->D * ((this->a - this->a_p) * digamma(this->a) - this->a
     - this->a_p * log(this->b_p) - lgamma(this->a) + lgamma(this->a_p))
     + this->b_p * this->a * this->ib.sum() + this->a_p * this->logb;
}
