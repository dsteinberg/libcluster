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

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

//TODO: make all protected variables private and accessed by protected functions
//      to improve encapsulation??

/*! Namespace that implements weight and cluster distributions. */
namespace distributions
{

//
// Namespace 'symbolic' constants
//

const double BETAPRIOR   = 1.0;      //!< beta prior value (Gaussians)
const double NUPRIOR     = 1.0;      //!< nu prior value (diagonal Gaussians)
const double ALPHA1PRIOR = 1.0;      //!< alpha1 prior value (All weight dists)
const double ALPHA2PRIOR = 1.0;      //!< alpha2 prior value (SB & Gdir)
const double APRIOR      = 1.0;      //!< a prior value (Exponential)


//
// Useful Typedefs
//

typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb; //!< Boolean Array


//
// Weight Parameter Distribution classes
//

/*! \brief To make a new weight class that will work with the algorithm
 *         templates, your class must have this as the minimum interface.
 */
class WeightDist
{
public:

  // WeightDist(), required inherited constructor template

  /*! \brief Update the distribution.
   *  \param Nk an array of observations counts.
   */
  virtual void update (const Eigen::ArrayXd& Nk) = 0;

  /*! \brief Evaluate the expectation of the log label weights in the mixtures.
   *  \returns An array of likelihoods for the labels given the weights
   */
  virtual const Eigen::ArrayXd& Elogweight () const = 0;

  /*! \brief Get the number of observations contributing to each weight.
   *  \returns An array the number of observations contributing to each weight.
   */
  const Eigen::ArrayXd& getNk () const { return this->Nk; }

  /*! \brief Get the free energy contribution of these weights.
   *  \returns the free energy contribution of these weights
   */
  virtual double fenergy () const = 0;

  /*! \brief virtual destructor.
   */
  virtual ~WeightDist() {}

protected:

  /*! \brief Default constructor to set an empty observation array.
   */
  WeightDist () : Nk(Eigen::ArrayXd::Zero(1)) {}

  Eigen::ArrayXd Nk; //!< Number of observations making up the weights.
};


/*!
 *  \brief Stick-Breaking (Dirichlet Process) parameter distribution.
 */
class StickBreak : public WeightDist
{
public:

  StickBreak ();

  StickBreak (const double concentration);

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Elogweight () const { return this->E_logpi; }

  double fenergy () const;

  virtual ~StickBreak () {}

protected:

  // Prior hyperparameters, expectations etc
  double alpha1_p;  //!< First prior param \f$ Beta(\alpha_1,\alpha_2) \f$
  double alpha2_p;  //!< Second prior param \f$ Beta(\alpha_1,\alpha_2) \f$
  double F_p;       //!< Free energy component dependent on priors only

  // Posterior hyperparameters and expectations
  Eigen::ArrayXd alpha1; //!< First posterior param corresp to \f$ \alpha_1 \f$
  Eigen::ArrayXd alpha2; //!< Second posterior param corresp to \f$ \alpha_2 \f$
  Eigen::ArrayXd E_logv; //!< Stick breaking log expectation
  Eigen::ArrayXd E_lognv; //!< Inverse stick breaking log expectation
  Eigen::ArrayXd E_logpi; //!< Expected log weights

  // Order tracker
  std::vector< std::pair<int,double> > ordvec; //!< For order specific updates

private:

  // Do some prior free energy calcs
  void priorfcalc (void);
};


/*!
 *  \brief Generalised Dirichlet parameter distribution (truncated stick
 *         breaking).
 */
class GDirichlet : public StickBreak
{
public:

  void update (const Eigen::ArrayXd& Nk);

  double fenergy () const;

  virtual ~GDirichlet () {}

};


/*!
 *  \brief Dirichlet parameter distribution.
 */
class Dirichlet : public WeightDist
{
public:

  Dirichlet ();

  Dirichlet (const double alpha);

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Elogweight () const { return this->E_logpi; }

  double fenergy () const;

  virtual ~Dirichlet () {}

private:

  // Prior hyperparameters, expectations etc
  double alpha_p; // Symmetric Dirichlet prior \f$ Dir(\alpha) \f$
  double F_p;     // Free energy component dependent on priors only

  // Posterior hyperparameters and expectations
  Eigen::ArrayXd alpha;   // Posterior param corresp to \f$ \alpha \f$
  Eigen::ArrayXd E_logpi; // Expected log weights

};


//
// Cluster Parameter Distribution classes
//

/*! \brief To make a new cluster distribution class that will work with the
 *         algorithm templates your class must have this as the minimum
 *         interface.
 */
class ClusterDist
{
public:

  /*! \brief Add observations to the cluster without updating the parameters
   *         (i.e. add to the sufficient statistics)
   *  \param qZk the observation indicators for this cluster, corresponding to
   *              X.
   *  \param X the observations [obs x dims], to add to this cluster according
   *           to qZk.
   */
  virtual void addobs (
      const Eigen::VectorXd& qZk,
      const Eigen::MatrixXd& X
      ) = 0;

  /*! \brief Update the cluster parameters from the observations added from
   *         addobs().
   */
  virtual void update () = 0;

  /*! \brief Clear the all parameters and observation accumulations from
   *         addobs().
   */
  virtual void clearobs () = 0;

  /*! \brief Evaluate the log marginal likelihood of the observations.
   *  \param X a matrix of observations, [obs x dims].
   *  \returns An array of likelihoods for the observations given this dist.
   */
  virtual Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const = 0;

  /*! \brief Get the free energy contribution of these cluster parameters.
   *  \returns the free energy contribution of these cluster parameters.
   */
  virtual double fenergy () const = 0;

  /*! \brief Propose a split for the observations given these cluster parameters
   *  \param X a matrix of observations, [obs x dims], to split.
   *  \returns a binary array of split assignments.
   *  \note this needs to consistently split observations between multiple
   *        subsequent calls, but can change after each update().
   */
  virtual ArrayXb splitobs (const Eigen::MatrixXd& X) const = 0;

  /*! \brief Return the number of observations belonging to this cluster.
   *  \returns the number of observations belonging to this cluster.
   */
  double getN () const { return this->N; }

  /*! \brief Return the cluster prior value.
   *  \returns the cluster prior value.
   */
  double getprior () const { return this->prior; }

  /*! \brief virtual destructor.
   */
  virtual ~ClusterDist() {}

protected:

  /*! \brief Constructor that must be called to set the prior and cluster
   *         dimensionality.
   *  \param prior the cluster prior.
   *  \param D the dimensionality of this cluster.
   */
  ClusterDist (const double prior, const unsigned int D)
    : D(D), prior(prior), N(0) {}

  unsigned int D; //!< Dimensionality
  double prior;   //!< Cluster prior
  double N;       //!< Number of observations making up this cluster.

};


/*!
 *  \brief Gaussian-Wishart parameter distribution for full Gaussian clusters.
 */
class GaussWish : public ClusterDist
{
public:

  /*! \brief Make a Gaussian-Wishart prior.
   *
   *  \param clustwidth makes the covariance prior \f$ clustwidth \times D
   *          \times \mathbf{I}_D \f$.
   *  \param D is the dimensionality of the data
   */
  GaussWish (const double clustwidth, const unsigned int D);

  void addobs (const Eigen::VectorXd& qZk, const Eigen::MatrixXd& X);

  void update ();

  void clearobs ();

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;

  /*! \brief Get the estimated cluster mean.
   *  \returns the expected cluster mean.
   */
  const Eigen::RowVectorXd& getmean () const { return this->m; }

  /*! \brief Get the estimated cluster covariance.
   *  \returns the expected cluster covariance.
   */
  Eigen::MatrixXd getcov () const { return this->iW/this->nu; }

  virtual ~GaussWish () {}

private:

  // Prior hyperparameters etc
  double nu_p;
  double beta_p;
  Eigen::RowVectorXd m_p;
  Eigen::MatrixXd iW_p;
  double logdW_p;
  double F_p;

  // Posterior hyperparameters
  double nu;              // nu, Lambda ~ Wishart(W, nu)
  double beta;            // beta, mu ~ Normal(m, (beta*Lambda)^-1)
  Eigen::RowVectorXd m;   // m, mu ~ Normal(m, (beta*Lambda)^-1)
  Eigen::MatrixXd iW;     // Inverse W, Lambda ~ Wishart(W, nu)
  double logdW;           // log(det(W))

  // Sufficient Statistics
  double N_s;
  Eigen::RowVectorXd x_s;
  Eigen::MatrixXd xx_s;

};


/*!
 *  \brief Normal-Gamma parameter distribution for diagonal Gaussian clusters.
 */
class NormGamma : public ClusterDist
{
public:

  /*! \brief Make a Normal-Gamma prior.
   *
   *  \param clustwidth makes the covariance prior \f$ clustwidth \times
   *         \mathbf{I}_D \f$.
   *  \param D is the dimensionality of the data
   */
  NormGamma (const double clustwidth, const unsigned int D);

  void addobs (const Eigen::VectorXd& qZk, const Eigen::MatrixXd& X);

  void update ();

  void clearobs ();

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;

  /*! \brief Get the estimated cluster mean.
   *  \returns the expected cluster mean.
   */
  const Eigen::RowVectorXd& getmean () const { return this->m; }

  /*! \brief Get the estimated cluster covariance.
   *  \returns the expected cluster covariance (just the diagonal elements).
   */
  Eigen::RowVectorXd getcov () const { return this->L*this->nu; }

  virtual ~NormGamma () {}

private:

  // Prior hyperparameters etc
  double nu_p;
  double beta_p;
  Eigen::RowVectorXd m_p;
  Eigen::RowVectorXd L_p;
  double logL_p;

  // Posterior hyperparameters
  double nu;
  double beta;
  Eigen::RowVectorXd m;
  Eigen::RowVectorXd L;
  double logL;

  // Sufficient Statistics
  double N_s;
  Eigen::RowVectorXd x_s;
  Eigen::RowVectorXd xx_s;

};


/*!
 *  \brief Exponential-Gamma parameter distribution for Exponential clusters.
 */
class ExpGamma : public ClusterDist
{
public:

  /*! \brief Make a Gamma prior.
   *
   *  \param obsmag is the prior value for b in Gamma(a, b), which works well
   *                when it is approximately the magnitude of the observation
   *                dimensions, x_djn.
   *  \param D is the dimensionality of the data
   */
  ExpGamma (const double obsmag, const unsigned int D);

  void addobs (const Eigen::VectorXd& qZk, const Eigen::MatrixXd& X);

  void update ();

  void clearobs ();

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;

  /*! \brief Get the estimated cluster rate parameter, i.e. Exp(E[lambda]),
   *         where lambda is the rate parameter.
   *  \returns the expected cluster rate parameter.
   */
  Eigen::RowVectorXd getrate () { return this->a*this->ib; }

  virtual ~ExpGamma () {}

private:

  // Prior hyperparameters
  double a_p;
  double b_p;

  // Posterior hyperparameters etc
  double a;
  Eigen::RowVectorXd ib;  // inverse b
  double logb;

  // Sufficient Statistics
  double N_s;
  Eigen::RowVectorXd x_s;

};


}

#endif // DISTRIBUTIONS_H
