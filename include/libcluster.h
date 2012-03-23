#ifndef LIBCLUSTER_H
#define LIBCLUSTER_H

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <vector>


//
// Namespace Definitions
//

/*! \brief Namespace that contains implementations of Bayesian mixture model
 *         based algorithms for clustering.
 *
 *  This namespace provides various Bayesian mixture models that can be used
 *  for clustering data. The algorithms that have been implemented are:
 *
 *    - Variational Dirichlet Process (VDP) for Gaussian observations [1].
 *    - The Bayesian Gaussian Mixture model [4] ch 11.
 *    - Groups of Mixtures Clustering (GMC) model for Gaussian observations [3].
 *    - Symmetric Groups of Mixtures Clustering (S-GMC) model for Gaussian
 *      observations [3].
 *
 *  All of these algorithms infer the number of clusters present in the data.
 *  Also, the sufficient statistics of the clusters are returned, which can be
 *  subsequently used to initialise these models for incremental clustering.
 *
 * [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
 *     Dirichlet process mixtures, Advances in Neural Information Processing
 *     Systems, vol. 19, p. 761, 2007.
 *
 * [2] Y. Teh, K. Kurihara, and M. Welling. Collapsed variational inference
 *     for HDP. Advances in Neural Information Processing Systems,
 *     20:1481–1488, 2008.
 *
 * [3] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Clustering Groups of
 *     Related Visual Datasets," unpublished, 2011.
 *
 * [4] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
 *     Springer Science+Business Media, 2006.
 *
 * \author Daniel Steinberg
 *         Australian Centre for Field Robotics
 *         The University of Sydney
 *
 * \date   23/03/2012
 */
namespace libcluster
{

//
// Namespace constants (use as argument defaults)
//

const double PRIORVAL = 1e-5;  //!< Default prior hyperparameter value


//
// The Sufficient Statistics container class (suffstat.cpp)
//

/*! \brief Sufficient statistics container class.
 *
 *  This class stores the cluster parameter sufficient statistics from the
 *  various clustering algorithms in this library. Most exponential family
 *  distributions can be created from sufficient statistics, and so the cluster
 *  parameters (for various distributions) can be directly computed from this
 *  class.
 *
 *  By storing sufficient statistics, and also free energy contributions, this
 *  class facilitates incremental/online clustering.
 */
class SuffStat
{
public:

  /*! \brief SuffStat constructor.
   *  \param prior the prior value to use with the clustering model. This is
   *         used to construct cluster distributions, and also for checking the
   *         compatibility of these suff. stats. with other suff. stats.
   */
  SuffStat (double prior = PRIORVAL);

  /*! \brief Set the sufficient statistics directly.
   *  \param k the cluster number
   *  \param N the number of observations in the cluster
   *  \param suffstat1 the first sufficient statistic.
   *  \param suffstat2 the second sufficient statistic.
   *  \throws invalid_argument if incompatible sufficient statists are detected.
   */
  void setSS (
      unsigned int k,
      double N,
      const Eigen::MatrixXd& suffstat1,
      const Eigen::MatrixXd& suffstat2
      );

  /*! \brief Set the free energy contribution of these sufficient statistics
   *  \param Fcontrib free energy contribution of these sufficient statistics
   */
  void setF (double Fcontrib) { this->F = Fcontrib; }

  /*! \brief Get the number of clusters represented by this object.
   *  \returns the number of clusters.
   */
  unsigned int getK () const { return this->K; }

  /*! \brief Get the free energy corresponding to these sufficient stats.
   *  \returns the free energy.
   */
  double getF () const { return this->F; }

  /*! \brief Get the cluster prior value used for cluster creation.
   *  \returns the prior value.
   */
  double getprior () const { return this->priorval; }

  /*! \brief Get the number of observations summarised by these sufficient
   *         stats. for a specific cluster.
   *  \param k the cluster to get the observation count from.
   *  \returns the observations count.
   *  \throws invalid_argument if k < 0 or k >= K.
   */
  double getN_k (unsigned int k) const;

  /*! \brief Get the first sufficient statistic corresponding to a particular
   *         cluster
   *  \param k the cluster to get the first suff. stat. from.
   *  \returns the first suff. stat. from cluster k.
   *  \throws invalid_argument if k < 0 or k >= K.
   */
  const Eigen::MatrixXd& getSS1 (unsigned int k) const;

  /*! \brief Get the second sufficient statistic corresponding to a particular
   *         cluster
   *  \param k the cluster to get the second suff. stat. from.
   *  \returns the second suff. stat. from cluster k.
   *  \throws invalid_argument if k < 0 or k >= K.
   */
  const Eigen::MatrixXd& getSS2 (unsigned int k) const;

  /*! \brief Add other sufficient statistics to these sufficient statistics.
   *  \param SS another sufficient statistic object to add to this one.
   *  \throws invalid_argument if incompatible sufficient statists are detected.
   */
  void addSS (const SuffStat& SS);

  /*! \brief Subtract other sufficient statistics from these suff. stats.
   *  \param SS another sufficient statistic object to subtract from this one.
   *  \throws invalid_argument if incompatible sufficient statists are detected.
   */
  void subSS (const SuffStat& SS);

  /*! \brief Add other sufficient statistics free energy contributions to these.
   *  \param SS another sufficient statistic object to add to this one.
   *  \throws invalid_argument if incompatible sufficient statists are detected.
   */
  void addF (const SuffStat& SS);

  /*! \brief Subtract other sufficient statistics free energy contributions from
   *         these.
   *  \param SS another sufficient statistic object to subtract from this one.
   *  \throws invalid_argument if incompatible sufficient statists are detected.
   */
  void subF (const SuffStat& SS);

  /*! \brief Virtual destructor to ensure derived classes clean up properly */
  virtual ~SuffStat () {}

private:

  bool compcheck (const SuffStat& SS);

  unsigned int K;   // Number of clusters
  double F;         // Contribution to modelf Free energy.
  double priorval;  // Prior value associated with suff. stats

  std::vector<double> N_k;              // The number of obs. in cluster k
  std::vector<Eigen::MatrixXd> SS1;     // Suff. Stat. 1
  std::vector<Eigen::MatrixXd> SS2;     // Suff. Stat. 2

};


/*! \brief Sufficient statistic stream operator.
 *
 *  The output of this operator will look something like:
 *
 *  \verbatim
 *    K = 2
 *    Nk = [19.1 30.9]
 *
 *    Suff. Stat. 1(0) =
 *      <Eigen Matrix Type>
 *    Suff. Stat. 1(1) =
 *      <Eigen Matrix Type>
 *    Suff. Stat. 2(0) =
 *      <Eigen Matrix Type>
 *    Suff. Stat. 2(1) =
 *      <Eigen Matrix Type>
 *  \endverbatim
 *
 *  \param s the stream to stream the suff. stat. object to.
 *  \param SS the sufficient statistic object.
 *  \returns a stream with SS.
 */
std::ostream& operator<< (std::ostream& s, const SuffStat& SS);


//
// Mixture Models for Clustering
//

/*! \brief The learning algorithm for the Variational Dirichlet Process for
 *         Gaussian clusters.
 *
 * This function implements the VDP clustering algorithm as specified by
 * [1], however a different 'nesting' strategy is used. The nesting strategy
 * sets all q(z_n > K) = 0, rather than setting the parameter distributions
 * equal to their priors over this truncation bound, K. This is the same
 * nesting strategy as used in [2].
 *
 *  \param X the observation matrix, NxD where N is the number of
 *         observations, and D is the number of dimensions.
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param SS a mutable SuffStat object to store the sufficient statistics of
 *         the clusters.
 *  \param diagcov use diagonal covariance? Default false, use Gaussian-Wishart
           priors, or true, use Normal-Gamma priors.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param ostrm stream to print notifications to. Defaults to std::cout.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy, unexpected empty
 *          clusters etc.
 *
 *  \note if you already have SS from a previous model, this can be used to
 *        initialise this model for incremental clustering.
 */
double learnVDP (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    libcluster::SuffStat& SS,
    const bool diagcov = false,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );


/*! \brief The learning algorithm for a Bayesian Gaussian Mixture model.
 *
 * This function implements the Bayseian GMM clustering algorithm as specified
 * by [1]. This algorithm may find a sub-optimal number of clusters as the only
 * the splitting heusristic used by [1] is implemented. In practice I have found
 * this performs almost identically to the VDP.
 *
 *  \param X the observation matrix, NxD where N is the number of
 *         observations, and D is the number of dimensions.
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param SS a mutable SuffStat object to store the sufficient statistics of
 *         the clusters.
 *  \param diagcov Use diagonal covariance? Default false, use Gaussian-Wishart
           priors, or true, use Normal-Gamma priors.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param ostrm stream to print notifications to. Defaults to std::cout.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy, unexpected empty
 *          clusters etc.
 *
 *  \note if you already have SS from a previous model, this can be used to
 *        initialise this model for incremental clustering.
 */
double learnGMM (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    libcluster::SuffStat& SS,
    const bool diagcov = false,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );


/*! \brief The learning algorithm for the Groups of Mixtures Clustering model.
 *
 * This function implements the Groups of Mixtues Clustering model algorithm
 * as specified by [3], with the additional of a "sparse" option. The GMC uses a
 * Generalised Dirichlet prior on the group mixture weights.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions.
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param SSgroups a mutable SuffStat object to store the sufficient statistics
 *         of the group cluster contributions.
 *  \param SS a mutable SuffStat object to store the sufficient statistics of
 *         the model clusters.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param diagcov Use diagonal covariance? Default false, use Gaussian-Wishart
           priors, or true, use Normal-Gamma priors.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param ostrm stream to print notifications to. Defaults to std::cout.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy, unexpected empty
 *          clusters etc.
 *
 *  \note if you already have SS and optinally SSgroups from a previous model,
 *        these can be used to initialise this model for incremental clustering.
 */
double learnGMC (
    const std::vector<Eigen::MatrixXd>& X,
    std::vector<Eigen::MatrixXd>& qZ,
    std::vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse = false,
    const bool diagcov = false,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );


/*! \brief The learning algorithm for the Symmetric Groups of Mixtures
 *         Clustering model.
 *
 * This function implements the Symmetric Groups of Mixtures Clustering model
 * as specified by [3], with the additional of a "sparse" option. The Symmetric
 * GMC uses a symmetric Dirichlet prior on the group mixture weights.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions.
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param SSgroups a mutable SuffStat object to store the sufficient statistics
 *         of the group cluster contributions.
 *  \param SS a mutable SuffStat object to store the sufficient statistics of
 *         the model clusters.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param diagcov Use diagonal covariance? Default false, use Gaussian-Wishart
           priors, or true, use Normal-Gamma priors.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param ostrm stream to print notifications to. Defaults to std::cout.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy, unexpected empty
 *          clusters etc.
 *
 *  \note if you already have SS and optinally SSgroups from a previous model,
 *        these can be used to initialise this model for incremental clustering.
 */
double learnSGMC (
    const std::vector<Eigen::MatrixXd>& X,
    std::vector<Eigen::MatrixXd>& qZ,
    std::vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse = false,
    const bool diagcov = false,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );

}

#endif // LIBCLUSTER_H
