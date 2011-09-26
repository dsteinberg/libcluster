//  TODO:
//  - Still look to see if there is a better way to have groups evenly
//    contribute to the I-GMC like the online LDA method of Hoffman.
//  - Change reference [3]
//

#ifndef LIBCLUSTER_H
#define LIBCLUSTER_H

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <vector>


//
// Namespace Definitions
//

/*! \brief Namespace that contains implementations of various Gaussian based
 *         Bayesian non-parametric algorithms.
 *
 *  This namespace provides a Gaussian mixture model (GMM) class, that can be
 *  learned by the Variational Dirichlet Process (VDP) [1], or the Grouped
 *  Mixtures Clustering (GMC) model [3], specifically derived for Gaussians. It
 *  also contains an incremental version of the GMC (I-GMC) which uses a derived
 *  GMM class to keep track of various sufficient statistics and tuning
 *  parameters.
 *
 *  The VDP and GMC and I-GMC automatically infer the number of clusters in a
 *  dataset, and the GMC and I-GMC can be used to learn multiple datasets that
 *  share the same clusters.
 *
 *  Functions are also provided to classify and find the probability of new
 *  observations from the learned GMM or IGMC classes.
 *
 * [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
 *     Dirichlet process mixtures, Advances in Neural Information Processing
 *     Systems, vol. 19, p. 761, 2007.
 *
 * [2] Y. Teh, K. Kurihara, and M. Welling. Collapsed variational inference
 *     for HDP. Advances in Neural Information Processing Systems,
 *     20:1481–1488, 2008.
 *
 * [3] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Hierarchal Bayesian
 *     mixtures for clustering multiple related datasets." NIPS 2011
 *     Submission, June 2011.
 *
 * [4] M. Sato, “Online model selection based on the variational bayes,” Neural
 *     Computation, vol. 13, no. 7, pp. 1649–1681, 2001.
 *
 * \author Daniel Steinberg
 *         Australian Centre for Field Robotics
 *         The University of Sydney
 *
 * \date   06/08/2011
 */
namespace libcluster
{

  //
  // Namespace constants (use as argument defaults)
  //

  const double BCLUSTWIDTH  = 0.01;  //!< Default prior cluster width (batch)
  const double ICLUSTWIDTH  = 1;     //!< Default prior cluster width (increm.)


  //
  // General GMM class and functions (gmm.cpp)
  //

  /*! \brief Gaussian mixture model (GMM) class. This class stores all of the
   *         The GMM parameters, and provides methods for manipulating them.
   *
   *  \see classify(), predict(), learnVDP() and learnGMC() methods for
   *       functions that use this class, and IGMC for derived classes.
   */
  class GMM
  {
  public:

    /*! \brief GMM default constructor. Creates an emtpy GMM object (K=D=0 and
     *         empty vectors).
     */
    GMM ();


    /*! \brief GMM constructor. Each entry in the vectors represents a
     *         Gaussian mixture component.
     *
     *  \param mu vector of 1xD means
     *  \param sigma vector of DxD covariances
     *  \param w vector of scalar weights
     *  \throws std::invalid_argument if the vectors are empty, or not of the
     *          same size.
     */
    GMM (
        const std::vector<Eigen::RowVectorXd>& mu,
        const std::vector<Eigen::MatrixXd>& sigma,
        const std::vector<double>& w
        );


    /*! \brief Get the dimensionality of the Gaussian mixture model.
     *
     *  \returns The dimensionality, D.
     */
    int getD () const { return this->D; }


    /*! \brief Get the number of mixtures in the Gaussian mixture model.
     *
     *  \returns The number of mixtures, K.
     */
    int getK () const { return this->K; }


    /*! \brief Get a Gaussian's covariance matrix.
     *
     *  \param k the k'th Gaussian to retrieve the covariance matrix from.
     *  \returns the DxD covariance matrix of the k'th Gaussian.
     *  \throws std::invalid_argument if k is not a valid cluster number.
     */
    const Eigen::MatrixXd& getsigma (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->sigma.empty() == true)
        throw std::invalid_argument("No parameters!");
      return this->sigma[k];
    }


    /*! \brief Get a Gaussian's mean vector.
     *
     *  \param k the k'th Gaussian to retrieve the mean vector from.
     *  \returns the 1xD mean vector of the k'th Gaussian.
     *  \throws std::invalid_argument if k is not a valid cluster number.
     */
    const Eigen::RowVectorXd& getmu (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->mu.empty() == true)
        throw std::invalid_argument("No parameters!");
      return this->mu[k];
    }


    /*! \brief Get a Gaussian's weight in the mixture.
     *
     *  \param k the k'th Gaussian to retrieve the weight of.
     *  \returns the scalar weight of the k'th Gaussian.
     *  \throws std::invalid_argument if k is not a valid cluster number.
     */
    double getw (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->w.empty() == true)
        throw std::invalid_argument("No parameters!");
      return this->w[k];
    }


  protected:

    unsigned int D;                              // Dimension of data
    unsigned int K;                              // Number of classes

    std::vector<Eigen::MatrixXd> sigma; // Mixture covariance matrices
    std::vector<Eigen::RowVectorXd> mu; // Mixture mean vectors
    std::vector<double> w;              // Mixture weights

  };


  /*! \brief Output stream operator for a GMM class object. e.g.
   *         stream << gmm << endl;
   *
   *  \param s the output stream.
   *  \param gmm the GMM object.
   *  \returns A textual summary of the Gaussian mixture's properties.
   */
  std::ostream& operator<< (std::ostream& s, const GMM& gmm);


  //
  // Functions that operate on the GMM (gmm.c)
  //

  /*! \brief Classify observations with a learned Gaussian mixture model.
   *
   *  \param X the observations to classify, this should be an NxD matrix.
   *  \param gmm a learned Gaussian mixture model object.
   *  \returns p(Z|X) NxK matrix of the probability of a label, z, given x.
   *  \throws std::invalid_argument if there are dimensionality
   *          inconsistencies or runtime errors
   */
  Eigen::MatrixXd classify (const Eigen::MatrixXd& X, const libcluster::GMM& gmm);


  /*! \brief Evaluate the predictive density for new observations from a
   *         learned Gaussian mixture model, p(X|pi, mu, sigma).
   *
   *  \param X the observations for which to evaluate the predicitve density,
   *         this should be an NxD matrix.
   *  \param gmm a learned Gaussian mixture model object.
   *  \returns p(X|pi, mu, sigma) an Nx1 vector of the probabilities of each x
   *           given the current GMM parameters. Each row is:
   *           p(X|pi, mu, sigma)) = sum_K pi_k*Norm(X*|mu_k, sigma_k).
   *  \throws std::invalid_argument if there are dimensionality
   *          inconsistencies or runtime errors
   */
  Eigen::VectorXd predict (const Eigen::MatrixXd& X, const libcluster::GMM& gmm);


  //
  // Variational Dirichlet Process (VDP) model (vdp.cpp)
  //

  /*! \brief The learning algorithm for the Variational Dirichlet Process.
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
   *  \param gmm a mutable GMM object to store the learned parameters of the
   *         clustering model.
   *  \param verbose flag for triggering algorithm status messages. Default is
   *         0 = silent.
   *  \param clustwidth our prior expectation of the "width" of a cluster.
   *         This is really a proportion of the first eigenvalue of the
   *         covariance of the data, a good range is between 0.01 and 1.
   *         The default is 0.01;
   *  \param ostrm stream to print notifications to. Defaults to std::cout.
   *  \returns Final free energy
   *  \throws std::logic_error if there are invalid argument calls such as
   *          non-PSD matrix calculations.
   *  \throws std::runtime_error if there are runtime issues with the VDP
   *          algorithm such as negative free energy, unexpected empty
   *          clusters etc.
   */
  double learnVDP (
      const Eigen::MatrixXd& X,
      Eigen::MatrixXd& qZ,
      libcluster::GMM& gmm,
      const bool verbose = false,
      const double clustwidth = BCLUSTWIDTH,
      std::ostream& ostrm = std::cout
      );

  //
  // Grouped Mixtues Clustering (GMC) model (gmc.cpp)
  //

  /*! \brief The learning algorithm for the Grouped Mixtues Clustering model.
   *
   * This function implements the Grouped Mixtues Clustering model clustering
   * algorithm as specified by [3], with the additional of a "sparse" option.
   *
   *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
   *         the number of observations in each group, j, and D is the number
   *         of dimensions.
   *  \param qZ is a vector of N_jxK matrices of the variational posterior
   *         approximations to p(z_j|X_j). K is the number of model clusters.
   *         This will always be overwritten to start with one cluster.
   *  \param w is a vector of cluster weights per group, j. This indicates the
   *         proportion of the custers in each group of data, X_j.
   *  \param gmm a mutable GMM object to store the K learned parameters of the
   *         entire clustering model (no distinguishing between groups).
   *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
   *         small amount of accuracy is traded off for a potentially large
   *         speed increase by not updating zero group weight cluster
   *         observation likelihoods. By default this is not enabled.
   *  \param verbose flag for triggering algorithm status messages. Default is
   *         0 = silent.
   *  \param clustwidth our prior expectation of the "width" of a cluster.
   *         This is really a proportion of the first eigenvalue of the
   *         covariance of the data, a good range is between 0.01 and 1.
   *         The default is 0.01;
   *  \param ostrm stream to print notifications to. Defaults to std::cout.
   *  \returns Final free energy
   *  \throws std::logic_error if there are invalid argument calls such as
   *          non-PSD matrix calculations.
   *  \throws std::runtime_error if there are runtime issues with the GMC
   *          algorithm such as negative free energy, unexpected empty
   *          clusters etc.
   */
  double learnGMC (
      const std::vector<Eigen::MatrixXd>& X,
      std::vector<Eigen::MatrixXd>& qZ,
      std::vector<Eigen::RowVectorXd>& w,
      libcluster::GMM& gmm,
      const bool sparse = false,
      const bool verbose = false,
      const double clustwidth = BCLUSTWIDTH,
      std::ostream& ostrm = std::cout
      );


  //
  // Incremental Grouped Mixtues Clustering (I-GMC) model (igmc.cpp)
  //

  /*! \brief Incremental Grouped Mixture Clustering model (I-GMC) class. This
   *         class stores all of the I-GMC sufficient statistics, discount and
   *         learning rates, prior GMC hyper parameters knobs and also the
   *         underlying GMM parameters. It provides methods for manipulating
   *         these properties.
   *
   *  \see learnIGMC() and classifyIGMC() methods that use this class.
   */
  class IGMC : public GMM
  {
  public:

    /*! \brief Main constructor for the I-GMC class. This sets various things
     *         that are used to calculate the discount and learning rates of the
     *         I-GMC, as well as knobs to tune some of the I-GMC prior
     *         hyperparameters.
     *
     *  \param J is the number of groups we expect to use in the calculation of
     *         the I-GMC.
     *  \param D is the dimensionality of the data that will be used. I.e. the
     *         number of columns of X.
     *  \param kappa influences the learning rate of the latter observations, if
     *         this is higher, the count less.
     *  \param tau0 is the effective number of prior observations, if this is
     *         higher, the first few observations will have more weight and be
     *         discounted less.
     *  \param cmeanp this is the prior expectation of the centre of the
     *         clusters. I usually just take the mean of the available data.
     *  \param cwidthp this is the prior expectation of the width of the
     *         clusters in absolute terms, i.e. setting this to 1 make the prior
     *         cluster covariance the identity matrix I_D.
     *  \throws std::invalid_argument if a bad value is passed in.
     */
    IGMC (
        unsigned int J,
        unsigned int D,
        double kappa,
        double tau0,
        const Eigen::RowVectorXd& cmeanp,
        double cwidthp = ICLUSTWIDTH
        );

    IGMC (const IGMC& igmc, unsigned int k); // Copy constructor for only one SS

    // Use this to add new groups, and reset the number of observations
    void setcounts (unsigned int J, unsigned int tau);

    unsigned int getJ () const { return this->J; }

    unsigned int gettau () const { return this->tau - 1; }

    const Eigen::RowVectorXd& getcmeanp (void) const { return this->cmeanp; }

    double getcwidthp () const { return this->cwidthp; }

    double getNk (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->Nk_.empty() == true)
        throw std::invalid_argument("No sufficient statistics!");
      return this->Nk_[k];
    }

    const Eigen::RowVectorXd& getxk (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->Xk_.empty() == true)
        throw std::invalid_argument("No sufficient statistics!");
      return this->Xk_[k];
    }

    const Eigen::MatrixXd& getRk (unsigned int k) const
    {
      if (k >= K)
        throw std::invalid_argument("Invalid k!");
      if (this->Rk_.empty() == true)
        throw std::invalid_argument("No sufficient statistics!");
      return this->Rk_[k];
    }

    void calcSS (
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& qZ,
        std::vector<double>& Nk,
        std::vector<Eigen::RowVectorXd>& Xk_,
        std::vector<Eigen::MatrixXd>& Rk_
        ) const;

    void calcF (double& Fpi, double& Fxz) const;


    // deletes all w, mu and sigma! TODO Fix this so it respects the params??
    void delSS (unsigned int k);

    void update (
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& qZ,
        const std::vector<double>& w,
        const std::vector<Eigen::RowVectorXd>& mu,
        const std::vector<Eigen::MatrixXd>& sigma,
        double& Fpi,
        double& Fxz
        );

  protected:  // Protected so we can copy these members in the matlab interface

    IGMC () {}

    unsigned int J;   // Total number of groups we are going to use
    unsigned int tau; // Number of observations/batches in the stats seen.

    double kappa;     // Asymptotic decreasing learning ratio constant.
    double tau0;      // Effective number of samples contributing to stats.
    double lambda;    // Learning discount factor
    double rho;       // Effective learning rate

    Eigen::RowVectorXd cmeanp; // Prior tuning param. for cluster mean location
    double cwidthp;            // Prior tuning parameter for cluster width

    double Fpi;       // Free energy contribution of the weights
    double Fxz;       // Free energy contribution of the complete data likelihd.
    std::vector<double> Nk_;               // Number of obs suff. stat.
    std::vector<Eigen::RowVectorXd> Xk_;   // Effective observations suff. stat.
    std::vector<Eigen::MatrixXd> Rk_;      // Obs. outer product suff. stat.

  };


  /*! \brief Output stream operator for an IGMC class object. e.g.
   *         stream << igmc << endl;
   *
   *  \param s the output stream.
   *  \param igmc the IGMC object.
   *  \returns A textual summary of the incremental Gaussian mixture's
   *           properties and current state of the IGMC.
   */
//  friend std::ostream& operator<< (std::ostream& s, const iGMC& igmc); // TODO


  /*! \brief The learning algorithm for the Incremental Grouped Mixtues
   *         Clustering model.
   *
   * This function implements an Incremental Grouped Mixtues Clustering model
   * clustering algorithm, which is based on an incremental VB approach from
   * [4], with the additional of a "sparse" option (TODO). Call this for every
   * group or batch of data AFTER creating an IGMC object.
   *
   *  \param X the observation matrix for the current group of data, NxD where N
   *         is the number of observations, and D is the number of dimensions.
   *  \param igmc a mutable IGMC object to store the K sufficient statistics,
   *         discount parameters and K GMM learned parameters of the
   *         entire clustering model (no distinguishing between groups).
   *  \param sparse flag for enabling the "sparse" updates for the I-GMC. Some
   *         small amount of accuracy is traded off for a potentially large
   *         speed increase by not updating zero group weight cluster
   *         observation likelihoods. By default this is not enabled.
   *  \param verbose flag for triggering algorithm status messages. Default is
   *         0 = silent.
   *  \param ostrm stream to print notifications to. Defaults to std::cout.
   *  \returns true if a call to this function has not significantly changed the
   *           IGMC object, false if there has been a significant change, in a
   *           Bayesian sense.
   *  \throws std::logic_error if there are invalid argument calls such as
   *          non-PSD matrix calculations.
   *  \throws std::runtime_error if there are runtime issues with the I-GMC
   *          algorithm such as negative free energy, non PSD or singular
   *          matrices etc.
   */
  bool learnIGMC (
    const Eigen::MatrixXd& X,
    libcluster::IGMC& igmc,
//    const bool sparse = false,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );


  /*! \brief Classify observations with a learned Incremental Grouped Mixtures
   *         Clustering model.
   *
   * This is an iterative algorithm that finds the variational posterior label
   * assignment distribution for this group of data, q(Z_j). It also calculates
   * the weights, pi_j for this group. This should really be called whenever the
   * I-GMC has been updated and you wish to get these properties for a group of
   * data.
   *
   * This algorithm essentially just runs the VBE update and VBM update for the
   * group mixture weights, while leaving the cluster parameters the same.
   *
   *  \param X the observations to classify, this should be an NxD matrix.
   *  \param igmc a learned I-GMC object.
   *  \param qZ p(Z|X) NxK matrix of the probability of a label, z, given x.
   *  \returns The GMC weights for this group of data, pi_j. I.e. the GMM that
   *           is exclusive to this group of data.
   *  \param verbose flag for triggering algorithm status messages. Default is
   *         0 = silent.
   *  \param ostrm stream to print notifications to. Defaults to std::cout.
   *  \throws std::invalid_argument if there are dimensionality
   *          inconsistencies or runtime errors
   */
  Eigen::RowVectorXd classifyIGMC (
    const Eigen::MatrixXd& X,
    const libcluster::IGMC& igmc,
    Eigen::MatrixXd& qZ,
    const bool verbose = false,
    std::ostream& ostrm = std::cout
    );
}

#endif // LIBCLUSTER_H
