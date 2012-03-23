#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

/*! Namespace that implements weight and cluster distributions. */
namespace distributions
{

//
// Useful Typedefs
//

typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb; //!< Boolean Array


//
// Namespace 'symbolic' constants
//

const double BETAPRIOR   = 1.0;      //!< beta prior value
const double NUPRIOR     = 1.0;      //!< nu prior value (for diagonal GMM)
const double ALPHA1PRIOR = 1.0;      //!< alpha1 prior value
const double ALPHA2PRIOR = 1.0;      //!< alpha2 prior value


//
// Weight Parameter Distribution classes
//

/*! \brief To make a new cluster weight class that will work with the algorithm
 *         templates your class must have this as the minimum interface.
 */
class WeightDist
{
public:

  // WeightDist(), required inherited contructor template

  /*! \brief Update the distribution.
   *  \param Nk an array of observations counts.
   */
  virtual void update (const Eigen::ArrayXd& Nk) = 0;

  /*! \brief Evaluate the log marginal likelihood of the labels.
   *  \returns An array of likelihoods for the labels given the weights
   */
  virtual const Eigen::ArrayXd& Eloglike () const = 0;

  /*! \brief Get the number of observations in each cluster.
   *  \returns An array the number of observations in each cluster.
   */
  virtual const Eigen::ArrayXd& getNk () const = 0;

  /*! \brief Get the free energy contribution of these weights.
   *  \returns the free energy contribution of these weights
   */
  virtual double fenergy () const = 0;

  /*! \brief virtual destructor.
   */
  virtual ~WeightDist() {}
};


/*!
 *  \brief Stick-Breaking (Dirichlet Process) parameter distribution.
 */
class StickBreak : public WeightDist
{
public:

  StickBreak ();

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Eloglike () const { return this->E_logpi; }

  const Eigen::ArrayXd& getNk () const { return this->Nk; }

  double fenergy () const;

  virtual ~StickBreak () {}

protected:

  // Prior hyperparameters, expectations etc
  double alpha1_p;
  double alpha2_p;
  double F_p;

  // Posterior hyperparameters and expectations
  Eigen::ArrayXd Nk;
  Eigen::ArrayXd alpha1;
  Eigen::ArrayXd alpha2;
  Eigen::ArrayXd E_logv;
  Eigen::ArrayXd E_lognv;
  Eigen::ArrayXd E_logpi;

  // Order tracker
  std::vector< std::pair<int,double> > ordvec;
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
};


/*!
 *  \brief Dirichlet parameter distribution.
 */
class Dirichlet : public WeightDist
{
public:

  Dirichlet ();

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Eloglike () const { return this->E_logpi; }

  const Eigen::ArrayXd& getNk () const { return this->Nk; }

  double fenergy () const;

  virtual ~Dirichlet () {}

protected:

  // Prior hyperparameters, expectations etc
  double alpha_p;
  double F_p;

  // Posterior hyperparameters and expectations
  Eigen::ArrayXd Nk;
  Eigen::ArrayXd alpha;
  Eigen::ArrayXd E_logpi;

};


//
// Cluster Parameter Distribution classes
//

/*! \brief To make a new cluster distribution class that will work with the
 *         algorithm templates your class must have this as the minimum
 *         interface.
 *
 *   In addition to the vitual member functions requiring implementation, the
 *   following STATIC member functions require definition:
 *
 *   Make sufficient statistics given observations X, and observation
 *   assignments q(Z = k), or qZk:
 *
 *    \code
 *      static void makeSS (
 *        const Eigen::VectorXd& qZk,
 *        const Eigen::MatrixXd& X,
 *        Eigen::MatrixXd& SuffStat1,
 *        Eigen::MatrixXd& SuffStat2
 *        );
 *    \endcode
 *
 *    Where qZk is the observation assignment probabilities of observations, X,
 *    to this cluster. SuffStat1 must return this clusters first sufficient
 *    statistic, and SuffStat2 must return the second.
 *
 *    Return the size of the sufficient statistics, given the observations X:
 *
 *    \code
 *      static Eigen::Array4i dimSS (const Eigen::MatrixXd& X);
 *    \endcode
 *
 *    Where the returning array must be of the form:
 *      [
 *        number of rows of SuffStat1,
 *        number of cols of SuffStat1,
 *        number of rows of SuffStat2,
 *        number of cols of SuffStat2
 *      ]
 *
 */
class ClusterDist
{
public:

  /*! \brief Update the distribution.
   *  \param N an array of observations counts belonging to this cluster
   *  \param suffstat1 sufficient statistic 1, made by makeSS()
   *  \param suffstat2 sufficient statistic 2, made by makeSS()
   */
  virtual void update (
      double N,
      const Eigen::MatrixXd& suffstat1,
      const Eigen::MatrixXd& suffstat2
      ) = 0;

  /*! \brief Evaluate the log marginal likelihood of the observations.
   *  \param X a matrix of observations, [obs, dims].
   *  \returns An array of likelihoods for the observations given this dist.
   */
  virtual Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const = 0;

  /*! \brief Get the free energy contribution of these cluster parameters.
   *  \returns the free energy contribution of these cluster parameters.
   */
  virtual double fenergy () const = 0;

  /*! \brief Propose a split for the observations given these cluster parameters
   *  \param X a matrix of observations, [obs, dims], to split.
   *  \returns a binary array of split assignments.
   *  \note this needs to consistently split observations between multiple
   *        subsequent calls, but can change after each update().
   */
  virtual ArrayXb splitobs (const Eigen::MatrixXd& X) const = 0;

  /*! \brief virtual destructor.
   */
  virtual ~ClusterDist() {}

protected:

  /*! \brief Constructor that must be called to set the prior and cluster
   *         dimensionality.
   *  \param prior the cluster prior.
   *  \param D the dimensionality of this cluster.
   */
  ClusterDist (const double prior, const unsigned int D) : D(D), prior(prior) {}

  unsigned int D;
  double prior;

};


/*!
 *  \brief Gaussian-Wishart parameter distribution for full Gaussian clusters.
 */
class GaussWish : public ClusterDist
{
public:

  /*! \brief Make an uninformed Gaussian-Wishart prior.
   *
   *  \param clustwidth makes the covariance prior \f$ clustwidth \times D
   *          \times \mathbf{I}_D \f$.
   *  \param D is the dimensionality of the data
   */
  GaussWish (const double clustwidth, const unsigned int D);

  static void makeSS (
      const Eigen::VectorXd& qZk,
      const Eigen::MatrixXd& X,
      Eigen::MatrixXd& x_s,       //!< [1xD] Row Vector sufficient stat.
      Eigen::MatrixXd& xx_s       //!< [DxD] Matrix sufficient stats.
      );

  static Eigen::Array4i dimSS (const Eigen::MatrixXd& X);

  void update (
        double N,
        const Eigen::MatrixXd& x_s, //!< [1xD] Row Vector sufficient stat.
        const Eigen::MatrixXd& xx_s //!< [DxD] Matrix sufficient stats.
      );

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;


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
  double N;
};

/*!
 *  \brief Normal-Gamma parameter distribution for diagonal Gaussian clusters.
 */
class NormGamma : public ClusterDist
{
public:

  /*! \brief Make an uninformed Normal-Gamma prior.
   *
   *  \param clustwidth makes the covariance prior \f$ clustwidth \times
   *         \mathbf{I}_D \f$.
   *  \param D is the dimensionality of the data
   */
  NormGamma (const double clustwidth, const unsigned int D);

  static void makeSS (
      const Eigen::VectorXd& qZk,
      const Eigen::MatrixXd& X,
      Eigen::MatrixXd& x_s,   //!< [1xD] Row Vector sufficient stat.
      Eigen::MatrixXd& xx_s   //!< [1xD] Row Vector sufficient stat.
      );

  static Eigen::Array4i dimSS (const Eigen::MatrixXd& X);

  void update (
        double N,
        const Eigen::MatrixXd& x_s, //!< [1xD] Row Vector sufficient stat.
        const Eigen::MatrixXd& xx_s //!< [1xD] Row Vector sufficient stat.
      );

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;


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
  double N;
};

}

#endif // DISTRIBUTIONS_H
