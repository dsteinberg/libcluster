#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "probutils.h"


/*! Namespace that implements weight and cluster distributions. */
namespace distributions
{


//
// Namespace 'symbolic' constants
//

const double BETAPRIOR   = 1.0;      //!< beta prior value
const double ALPHA1PRIOR = 1.0;      //!< alpha1 prior value
const double ALPHA2PRIOR = 1.0;      //!< alpha2 prior value


//
// Weight Parameter Distribution classes
//

/* To make a new distribution class that will work with the agorithm templates
 *  your class must have the following minimum interface:
 *
 *  class WeightDist
 *  {
 *  public:
 *
 *    void update (const Eigen::ArrayXd& Nk);
 *
 *    const Eigen::VectorXd& Emarginal () const;
 *
 *    const Eigen::ArrayXd& getNk () const
 *
 *    double fenergy () const;
 *
 *  };
 */

/*!
 *  \brief Stick-Breaking (Dirichlet Process) parameter distribution.
 */
class StickBreak
{
public:

  StickBreak ();

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Emarginal () const { return this->E_logpi; }

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
 *  \brief Generalised Dirichlet parameter distribution.
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
class Dirichlet
{
public:

  Dirichlet ();

  void update (const Eigen::ArrayXd& Nk);

  const Eigen::ArrayXd& Emarginal () const { return this->E_logpi; }

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

/* To make a new distribution class that will work with the algorithm templates
 *  your class must have the following minimum interface:
 *
 *  class ClusterDist
 *  {
 *  public:
 *
 *    void update (
 *      double N,
 *      const Eigen::RowVectorXd& x_s,
 *      const Eigen::MatrixXd& xx_s
 *    );
 *
 *    Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;
 *
 *    double fenergy () const;
 *
 *    ArrayXb splitobs (const Eigen::MatrixXd& X);
 *
 *    double getN () const
 *
 *  };
 *
 *  Notes:
 *   -  splitobs() needs to consistently split observations between multiple
 *      subsequent calls, but can change after each update().
 */

/*!
 *  \brief Gaussian-Wishart parameter distribution.
 */
class GaussWish
{
public:

  GaussWish (
      const double clustwidth,
      const Eigen::RowVectorXd& meanX,
      const Eigen::MatrixXd& covX
      );

  /*! \brief Make an uninformed Gaussian-Wishart prior.
   *
   *  \param cwidthp makes the covariance prior cwidthp*I.
   *  \param cmeanp is the mean prior cluster centre.
   *  \throws std::invalid_argument if cwidthp is less than or equal to 0.
   */
  GaussWish (const double cwidthp, const Eigen::RowVectorXd& cmeanp);

  void update (
        double N,
        const Eigen::RowVectorXd& x_s,
        const Eigen::MatrixXd& xx_s
      );

  Eigen::VectorXd Eloglike (const Eigen::MatrixXd& X) const;

  ArrayXb splitobs (const Eigen::MatrixXd& X) const;

  double fenergy () const;

  double getN () const { return this->N; }

  void getmeancov (Eigen::RowVectorXd& mean, Eigen::MatrixXd& cov) const;

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

}

#endif // DISTRIBUTIONS_H
