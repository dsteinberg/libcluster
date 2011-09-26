#ifndef VBCOMMON_H
#define VBCOMMON_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "probutils.h"


/*! Functions and classes that implement commonly occuring Variational
 *   Bayes hyperparameters, updates and other calculations for Gaussians,
 *   Stick Breaking entities (Generalised Dirichlet, DPs) etc.
 */
namespace vbcommon
{

  //
  // Namespace 'symbolic' constants
  //

  const int    SPLITITER   = 20;       //!< Max number of iter. for split VBEM
  const double CONVERGE    = 1.0e-5;   //!< Convergence threshold
  const double FENGYDEL    = CONVERGE/10; //!< Allowance for +ve F.E. steps
  const double BETAPRIOR   = 1.0;      //!< beta prior value
  const double ALPHA1PRIOR = 1.0;      //!< alpha1 prior value
  const double ALPHA2PRIOR = 1.0;      //!< alpha2 prior value


  //
  // Hyperparameter structures
  //

  /*!
   *  \brief Stick-Breaking hyperparameter class.
   */
  class SBparam
  {
  public:

    /*! \brief Get the alpha1 hyperparameter.
     *
     *  \returns alpha1
     */
    double getalpha1 () const { return this->alpha1; }

    /*! \brief Get the alpha2 hyperparameter.
     *
     *  \returns alpha2
     */
    double getalpha2 () const { return this->alpha2; }

    /*! \brief Get the expected log stick weight E[log(v)].
     *
     *  \returns E[log(v)]
     */
    double getE_logv () const { return this->E_logv; }

    /*! \brief Get the expected log 1 - stick weight E[log(1-v)].
     *
     *  \returns E[log(1-v)]
     */
    double getE_lognv () const { return this->E_lognv; }

  protected:

    // Stick-breaking Parmaeters
    double alpha1;   //!< alpha1, v_k ~ beta(alpha1, alpha2)
    double alpha2;   //!< alpha2, v_k ~ beta(alpha1, alpha2)

    // Log distribution expectations
    double E_logv;   //!< E[log(v_k)]
    double E_lognv;  //!< E[log(1-v_k)]
  };


  /*!
   *  \brief Stick-Breaking prior hyperparameter class.
   */
  class SBprior : public SBparam
  {
  public:

    /*! \brief Make an UNINFORMED Stick-breaking prior.
     *
     *  \param prior the prior Stick-breaking parameters (mutable).
     */
    SBprior ();

    /*! \brief Calculate the constant, prior only part of the Stick-breaking
     *         free energy.
     *  \returns the free energy.
     */
    double fnrgprior () const;

  };


  /*!
   *  \brief Stick-Breaking posterior hyperparameter class.
   */
  class SBposterior : public SBparam
  {
  public:

    /*! \brief Update the Stick-breaking Parameters using sufficient
     *         statistics.
     *
     *  \param Nk the number of observations belonging to this cluster.
     *  \param Ninf the number of observations belonging to all cluster
     *         SMALLER than this one.
     *  \param prior the prior Stick-breaking parameters.
     *  \note These parameters need to be updated in descending cluster size
     *        order.
     */
    void update (
        const double Nk,
        const double Ninf,
        const SBprior& prior
        );

    /*! \brief Calculate the posterior dependent part of the Stick-breaking
     *         free energy for one cluster.
     *
     *  \param prior the prior Stick-breaking parameters.
     *  \returns the free energy.
     */
    double fnrgpost (const SBprior& prior) const;

    /*! \brief Get the expectation of the log likelihood from a Discrete
     *         distribution, with respect to its parameters (SB prior).
     *
     *  \param cumE_lognv the accumulation of E[log(1-v_l) for all l < k in
     *         terms of cluster size order.
     *  \param tuncate use this flag to use a truncated stick-length for this
     *         evaluation. I.e. set v_K = 1. Defaults to false.
     *  \returns E[log(1-v_l)] NOT E_vk[log p(z=k|vk)], since this is useful
     *           to store, and can be accessed by the getE_logZ() method.
     *  \note Usually we call this as per the following pseudo snippet:
     *  \code
     *      ...
     *      double cumE_lognv = 0;
     *      for (each posterior in descending order, k)
     *      {
     *        cumE_lognv += posterior[k].Eloglike(cumE_lognvj);
     *      }
     *      double E_logZ = posterior[k].getE_logZ()
     *  \endcode
     *  \see getE_logZ()
     */
    double Eloglike (double cumE_lognv, bool truncate = false);

    /*! \brief Get the Expectation of log p(z|V) w.r.t. the variational
     *         posteriors
     *
     *  \returns E_vk[log p(z=k|vk)], using the variational posteriors for vk.
     */
    double getE_logZ () const { return this->E_logZ; }

  private:

    double E_logZ;   //!< E[log(q(Z))] basically the log expected weights
  };


  /*!
   *  \brief Gaussian-Wishart hyperparameter structure.
   */
  class GWparam
  {
  public:

    /*! \brief Get the nu hyperparameter for the Wishart distribution.
     *
     *  \returns nu, from Lambda = Wish(W, nu)
     */
    double getnu () const { return this->nu; }

    /*! \brief Get the beta hyperparameter for the Normal distribution.
     *
     *  \returns beta, from mu = Norm(m, (beta*Lambda)^-1)
     */
    double getbeta () const { return this->beta; }

    /*! \brief Get the mean hyperparameter for the Normal distribution.
     *
     *  \returns m, from mu = Norm(m, (beta*Lambda)^-1)
     */
    Eigen::RowVectorXd getm () const { return this->m; }

    /*! \brief Get a references to the mean hyperparameter for the Normal
     *         distribution.
     *
     *  \returns a reference to m, from mu = Norm(m, (beta*Lambda)^-1)
     */
    const Eigen::RowVectorXd& getrefm () const { return this->m; }

    /*! \brief Get the inverse W hyperparameter for the Wishart distribution.
     *
     *  \returns W^-1, from Lambda = Wish(W, nu)
     */
    Eigen::MatrixXd getiW () const { return this->iW; }

    /*! \brief Get a reference to the inverse W hyperparameter for the Wishart
     *         distribution.
     *
     *  \returns a reference to W^-1, from Lambda = Wish(W, nu)
     */
    const Eigen::MatrixXd& getrefiW () const { return this->iW; }

    /*! \brief Get the stored result of log(det(W))
     *
     *  \returns log(det(W))
     */
    double getlogdW () const { return this->logdW; }

  protected:

    // Parameters
    double nu;              //!< nu, Lambda ~ Wishart(W, nu)
    double beta;            //!< beta, mu ~ Normal(m, (beta*Lambda)^-1)
    Eigen::RowVectorXd m;   //!< m, mu ~ Normal(m, (beta*Lambda)^-1)
    Eigen::MatrixXd iW;     //!< Inverse W, Lambda ~ Wishart(W, nu)

    // Useful Statistics
    double logdW;           //!< log(det(W))
  };


  /*!
   *  \brief Gaussian-Wishart prior hyperparameter structure.
   */
  class GWprior : public GWparam
  {
  public:

    /*! \brief Make an uninformed Gaussian-Wishart prior.
     *
     *  \param cwidthp makes the covariance prior cwidthp*I.
     *  \param cmeanp is the mean prior cluster centre.
     *  \throws std::invalid_argument if cwidthp is less than or equal to 0.
     */
    GWprior (const double cwidthp, const Eigen::RowVectorXd& cmeanp);

    /*! \brief Make a semi-INFORMED Gaussian-Wishart prior.
     *
     *  \param clustwidth prior cluster width expectation.
     *  \param covX is the covariance of the observations, X.
     *  \param meanX is the mean of the observations, X.
     *  \throws std::invalid_argument if a non-PSD matrix is calculated.
     */
    GWprior (
        const double clustwidth,
        const Eigen::MatrixXd& covX,
        const Eigen::RowVectorXd& meanX
        );

    /*! \brief Calculate the constant, prior only part of the Gaussian-Wishart
     *         free energy.
     *  \returns the free energy.
     */
    double fnrgprior () const;

  };


  /*!
   *  \brief Gaussian-Wishart posterior hyperparameter structure.
   */
  class GWposterior : public GWparam
  {
  public:

    /*! \brief Update the Gaussian-Wishart Parameters using sufficient
     *         statistics
     *
     *  \param Nk the number of observations belonging to this cluster.
     *  \param xksum the unormalised sum of observations beloning to this
     *        cluster, xksum = sum (qz_n*x_n).
     *  \param Rksum the sum of the outer-products of the observations
     *         belonging to this cluster, Rksum = sum (qz_n*x_n*x_nT).
     *  \param prior the prior Gaussian-Wishart parameters.
     *  \throws std::invalid_argument if iW is not positive semidefinite.
     */
    void update (
        const double Nk,
        const Eigen::RowVectorXd& xksum,
        const Eigen::MatrixXd& Rksum,
        const GWprior& prior
        );

    /*! \brief Calculate the posterior dependent part of the Gaussian-Wishart
     *         free energy for one cluster.
     *
     *  \param prior the prior Gaussian-Wishart parameters.
     *  \returns the free energy.
     */
    double fnrgpost (const GWprior& prior) const;

    /*! \brief Get the expectation of the log likelihood from a Gaussian, with
     *         respect to the Gaussian parameters (GW prior).
     *
     *  \param X our observations [NxD].
     *  \returns expectation of the log Gaussian likelihood,
     *           E_mu,sigma[log N(x|mu, sigma)].
     *  \throws std::invalid_argument rethrown from the Mahalanobis distance
     *          evaluation.
     *  \throws std::runtime_error from digamma calculation (if nu_k <= 0).
     */
    Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

    /*! \brief Get the number of observations contributing to these posterior
     *         parameters.
     *
     *  \returns Nk, the number of observations making up this cluster's
     *           parameters.
     */
    double getNk () const { return this->Nk; }

  private:

    double Nk;    //!< Number of observations in this cluster
  };


  //
  // Functions
  //

  /*! \brief Partition the observations, X into only those with
   *         p(z_n = k|x_n) > 0.5.
   *
   *  \param X NxD matrix of observations.
   *  \param qZk Nx1 vector of probabilities p(z_n = k|x_n).
   *  \param Xk a MxD matrix of observations that belong to the cluster with
   *         p > 0.5 (mutable).
   *  \param map a Mx1 array of the locations of Xk in X (mutable).
   */
  void partX (
      const Eigen::MatrixXd& X,
      const Eigen::VectorXd& qZk,
      Eigen::MatrixXd& Xk,
      Eigen::ArrayXi& map
      );


  /*! \brief Augment the assignment matrix, qZ with the split cluster entry.
   *         The new cluster assignemnts are put in the K+1 th column in the
   *         return matrix
   *
   *  \param k the cluster to split (i.e. which column of qZ)
   *  \param map an index mapping from the array of split assignments to the
   *         observations in qZ. map[idx] -> n.
   *  \param split the boolean array of split assignments.
   *  \param qZ the [NxK] observation assignment probability matrix.
   *  \returns The new observation assignments, [Nx(K+1)].
   *  \throws std::invalid_argument if map.size() != split.size().
   */
  Eigen::MatrixXd augmentqZ (
      const double k,
      const Eigen::ArrayXi& map,
      const ArrayXb& split,
      const Eigen::MatrixXd& qZ
      );


  /*! Compare two pairs of type <int,double>.
   *
   *  \param i pair 1
   *  \param j pair 2
   *  \returns 1 if i.first > j.first
   */
  bool paircomp (
      const std::pair<int,double>& i,
      const std::pair<int,double> j
      );


  /*! Check if a vector of posterior distributions have any observations.
   *
   * \param postvec is a vector of posterior distribution classes. These classes
   *    must have a getNk() member function that returns a double (a count of
   *    observations).
   *  \returns True if any of the posterior distributions have observations
   *           associated with them, false otherwise.
   */
  template<class T> bool checkempty (const std::vector<T>& postvec)
  {
    for (unsigned int k = 0; k < postvec.size(); ++k)
      if (postvec[k].getNk() <= 1)
        return true;

    return false;
  }

}

#endif // VBCOMMON_H
