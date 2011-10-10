// TODO
// - This generally needs a lot more thought...
// - Make distribution agnostic interfaces like in the GMC and VDP
// - Make a sparse version!
// - Make the calcSS() interface a bit nicer, as well as some of the other IGMC
//   interfaces.
// - Find a nice way to not have to do Njk = qZ.colwise().sum() twice? calcSS()
//   and vbmaximisation();
// - Documentation

#include <limits>
#include "libcluster.h"
#include "probutils.h"
#include "distributions.h"
#include "vbcommon.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace distributions;
using namespace vbcommon;


//
// Public Member Functions
//

libcluster::IGMC::IGMC (
    unsigned int J,
    unsigned int D,
    double kappa,
    double tau0,
    const RowVectorXd& cmeanp,
    double cwidthp
    )
    : tau(1),
      lambda(0),
      rho(1),
      Fw(0),
      Fxz(0)
{
  if (kappa < 0)
    throw invalid_argument("kappa must be greater than or equal to 0!");
  if (tau0 < 1)
    throw invalid_argument("tau0 must be greater than or equal to 1!");
  if (J < 1)
    throw invalid_argument("J must be greater than or equal to 1!");
  if (D < 1)
    throw invalid_argument("D must be greater than or equal to 1!");
  if (cwidthp <= 0)
    throw invalid_argument("cwidthp must be greater than 0!");
  if(cmeanp.cols() != D)
    throw invalid_argument("cmeanp must have a length of D!");

  this->J = J;
  this->D = D;
  this->K = 0;
  this->kappa = kappa;
  this->tau0 = tau0;
  this->cwidthp = cwidthp;
  this->cmeanp = cmeanp;
}


libcluster::IGMC::IGMC (const IGMC& igmc, unsigned int k)
  : J(igmc.J),
    tau(igmc.tau),
    kappa(igmc.kappa),
    tau0(igmc.tau0),
    lambda(igmc.lambda),
    rho(igmc.rho),
    cmeanp(igmc.cmeanp),
    cwidthp(igmc.cwidthp),
    Fw(0),
    Fxz(0)
{
  this->D = igmc.D;

  // A few checks
  if ((igmc.K == 0) || (k >= igmc.K))
    return;     // There are no stored suff. stats. yet, so just return

  this->K = 1;
  this->N_s.push_back(igmc.N_s[k]);
  this->x_s.push_back(igmc.x_s[k]);
  this->xx_s.push_back(igmc.xx_s[k]);
}


void libcluster::IGMC::setcounts (unsigned int J, unsigned int tau)
{
  if ((J < 1) || (tau < 1))
    throw invalid_argument("J and tau must be greater than 0!");

  // Rescale the suff. stats.
  for (unsigned int k = 0; k < this->K; ++k)
  {
    this->N_s[k] = J * this->N_s[k] / this->J;
    this->x_s[k] = J * this->x_s[k] / this->J;
    this->xx_s[k] = J * this->xx_s[k] / this->J;
  }

  // Rescale the free energy contribs
  this->Fw = J * this->Fw / this->J;
  this->Fxz = J * this->Fxz / this->J;

  // Reset the obs counter and groups
  this->J   = J;
  this->tau = tau;

  // Reset discount and learning rates
  this->rho = 1;
  if (this->tau == 1)
    this->lambda = 0;
  else
  {
    // for  tau == 2
    this->lambda = 1 - 1 / this->tau0;
    this->rho    = 1 / (1 + this->lambda / this->rho);

    // for tau > 2, recursive
    for (unsigned int i = 2; i < this->tau; ++i)
    {
      this->lambda = 1 - (1-this->lambda) / ( 1+this->kappa * (1-this->lambda));
      this->rho    = 1 / (1 + this->lambda / this->rho);
    }
  }
}


void libcluster::IGMC::calcSS (
    const MatrixXd& X,
    const MatrixXd& qZ,
    vector<double>& Nk,
    vector<RowVectorXd>& xk,
    vector<MatrixXd>& Rk
    ) const
{
  unsigned int zK = qZ.cols();

  // Checks
  if (zK < (unsigned) this->K)
    throw invalid_argument("Invalid qZ! qZ has less columns than K!");
  if (X.cols() != this->D)
    throw invalid_argument("X must have the same dims as the suff. stats!");
  if ( (Nk.size() != zK) || (xk.size() != zK) || (Rk.size() != zK) )
    throw invalid_argument("Input vectors must be same length as qZ has cols!");

  // Initialise and precalculate stuff
  RowVectorXd Njk = qZ.colwise().sum();
  MatrixXd qZkX(X.rows(), this->D);

  // Now calculate the new suff. stats.
  for (unsigned int k = 0; k < zK; ++k)
  {
    qZkX = qZ.col(k).asDiagonal() * X;

    if (k < (unsigned) this->K)
    {
      Nk[k] = this->J*this->rho*Njk(k) + (1-this->rho)*this->N_s[k];
      xk[k] = this->J*this->rho*qZkX.colwise().sum()+(1-this->rho)*this->x_s[k];
      Rk[k] = this->J*this->rho*qZkX.transpose()*X + (1-this->rho)*this->xx_s[k];
    }
    else
    {
      Nk[k] = this->J*Njk(k);
      xk[k] = this->J*qZkX.colwise().sum();
      Rk[k] = this->J*qZkX.transpose()*X;
    }
  }
}


void libcluster::IGMC::calcF (double& Fpi, double& Fxz) const
{
  if (this->tau > 1)
  {
    Fpi = this->J*this->rho*Fpi + (1-this->rho)*this->Fw;
    Fxz = this->J*this->rho*Fxz + (1-this->rho)*this->Fxz;
  }
  else
  {
    Fpi = this->J*Fpi;
    Fxz = this->J*Fxz;
  }
}


void libcluster::IGMC::delSS (unsigned int k)
{
  if (k >= this->K)
    throw invalid_argument("Invalid mixture number in k, must be [0, K-1].");

  // Clear the GMM params because they will no longer be valid
  this->w.clear();
  this->mu.clear();
  this->sigma.clear();

  // Delete sufficient stats from sufficient stat vectors
  int delidx = k;
  this->N_s.erase(this->N_s.begin() + delidx);
  this->x_s.erase(this->x_s.begin() + delidx);
  this->xx_s.erase(this->xx_s.begin() + delidx);
  --this->K;
}


void libcluster::IGMC::update (
  const MatrixXd& X,
  const MatrixXd& qZ,
  const std::vector<double>& w,
  const std::vector<Eigen::RowVectorXd>& mu,
  const std::vector<Eigen::MatrixXd>& sigma,
  double& Fpi,
  double& Fxz
  )
{
  unsigned int zK = qZ.cols();

  // Checks
  if (zK < (unsigned) this->K)
    throw invalid_argument("Invalid qZ! qZ has less columns than K!");
  if (X.cols() != this->D)
    throw invalid_argument("X must have the same dims as the suff. stats!");
  if ( (w.size() != zK) || (mu.size() != zK) || (sigma.size() != zK) )
    throw invalid_argument("Input vectors must be same length as qZ has cols!");

  // Make temp arrays
  vector<double> Nk(zK);
  vector<RowVectorXd> xk(zK);
  vector<MatrixXd> Rk(zK);

  // Get new sufficient statistic and new Free energy calculations
  this->calcSS(X, qZ, Nk, xk, Rk);
  this->calcF(Fpi, Fxz);

  // Update sufficient stats
  this->N_s = Nk;
  this->x_s = xk;
  this->xx_s = Rk;
  this->K   = zK;

  // Update Free energy
  this->Fw = Fpi;
  this->Fxz = Fxz;

  // Update GMM parameters
  this->w = w;
  this->mu = mu;
  this->sigma = sigma;

  // Update discount factor
  if (this->tau > 2)        // Recursive discount formula
    this->lambda = 1 - (1-this->lambda) / (1 + this->kappa * (1-this->lambda));
  else                      // Calculate initial discount
    this->lambda = 1 - 1 / this->tau0;

  // Update learning rate
  this->rho = 1 / (1 + this->lambda / this->rho);

  ++this->tau;  // Increment observation count
}


//
// Private Functions
//

/* Variational Bayes Maximisation step for the I-GMC parameters.
 *  mutable: wdist, the weight parameter distribution.
 *  mutable: cdists, model cluster parameter distributions.
 */
template <class W, class C> void vbmaximisation (
    const MatrixXd& X,              // Observations in group J
    const MatrixXd& qZ,             // Observations to model mixture assignments
    const libcluster::IGMC& igmc,   // The current I-GMC object
    W& wdist,                       // Group weight parameter distribution
    vector<C>& cdists               // Cluster parameter distributions
//    const bool sparse             // Do sparse updates to groups
    )
{
  int K  = qZ.cols();

  // Calculate the new suff. stats. for the cluster parameter distributions
  vector<double> N_k(K);
  vector<RowVectorXd> x_k(K);
  vector<MatrixXd> xx_k(K);
  igmc.calcSS(X, qZ, N_k, x_k, xx_k);

  // Update the cluster parameter distributions
  for (int k = 0; k < K; ++k)
    cdists[k].update(N_k[k], x_k[k], xx_k[k]);

  // Update the cluster weight parameter distribution
  RowVectorXd Njk = qZ.colwise().sum();
  wdist.update(Njk);
}


/* The Variational Bayes Expectation step for a group of data.
 *  mutable: spost, vector of group posterior variational parameters
 *  mutable: Assignment probabilities, qZ
 *  returns: The complete-data (X,Z) free energy E[log p(X,Z)/q(Z)] for group j.
 *  throws: invalid_argument rethrown from other functions.
 *  NOTE: no discount to F is applied.
 */
template <class W, class C> double vbexpectation (
    const MatrixXd& X,
    MatrixXd& qZ,              // Observations to group mixture assignments
    const W& wdist,           // Group weight parameter distribution
    const vector<C>& cdists   // Cluster parameter distributions
//    const bool sparse         // Do sparse updates to groups
    )
{
  int K = qZ.cols(),
      Nj = X.rows();

  // Calculate expected log weights
  const ArrayXd E_logZ = wdist.Emarginal();

  // Expectations of cluster log likelihood for this group
  MatrixXd logqZ(Nj, K);
  for (int k = 0; k < K; ++k)
    logqZ.col(k) = E_logZ(k) + cdists[k].Eloglike(X).array();

  // Log normalisation constant of log observation likelihoods
  VectorXd logZzj = logsumexp(logqZ);

  // Compute Responsabilities
  qZ = ((logqZ.colwise() - logZzj).array().exp()).matrix();

  return -logZzj.sum();
}



/* Calculates the free energy lower bound for the model parameters.
 *  returns: the free energy of the parameter distributions
 */
template <class W, class C> void fenergy (
    const W& wdist,             // Weight parameter distribution
    const vector<C>& cdists,    // Cluster parameter distributions
    double& F_wj,               // Free energy contribution of weight params
    double& F_c                 // Free energy contribution of cluster params
    )
{
  // Calculate the cluster parameter free energy terms
  F_c = 0;
  for (unsigned int k = 0; k < cdists.size(); ++k)
    F_c += cdists[k].fenergy();

  // Calculate the weight parameter free energy terms
  F_wj = wdist.fenergy();
}


/* Incremental Varational Bayes EM for a group in the I-GMC.
 *  returns: the number of iterations of VB performed.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: wdist, the weight parameter distribution.
 *  mutable: cdists, model cluster parameter distributions.
 *  throws: invalid_argument from other functions or if cdists.size() does not
 *          match qZ.cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class C> int vbem (
    const MatrixXd& X,            // Observations of the current group
    MatrixXd& qZ,                 // Group obs. to model mixture assignments
    const libcluster::IGMC& igmc, // The current I-GMC object
    W& wdist,                     // Group weight parameter distribution
    vector<C>& cdists,            // Cluster parameter distributions
    double& F,                    // Free energy structure
    double& F_wj,                 // Free energy contribution of weight params
    double& F_xzj,                // Free energy contribution of data
    const int maxit = -1,         // Max 'meta' iterations (-1 = none, default)
//    const bool sparse,          // Do sparse updates to groups (default false)
    const bool verbose = false,   // Verbose output (default false)
    ostream& ostrm = cout         // Stream to print notification (cout default)
    )
{
  unsigned int K = qZ.cols();

  // Make sure cluster posterior vectors is right size
  if (cdists.size() != K)
    throw invalid_argument("Wrong number of cluster parameter distributions!");

  // Pre-allocate
  int i = 0;
  double Fold, F_c, dF_wj, dF_xzj;
  F = numeric_limits<double>::max();

  do
  {
    Fold = F;

    // VBM
    vbmaximisation<W, C>(X, qZ, igmc, wdist, cdists);

    // VBE for this group
    F_xzj = vbexpectation<W, C>(X, qZ, wdist, cdists);

    // Calculate the free energy of the model
    fenergy<W, C>(wdist, cdists, F_wj, F_c);
    dF_wj = F_wj;
    dF_xzj = F_xzj;
    igmc.calcF(dF_wj, dF_xzj); // Add in discount factors for Fpi and Fxz
    F = F_c + dF_wj + dF_xzj;

    ++i;
    if (F < 0)                    // Check for bad free energy calculation
      throw runtime_error("Calculated a negative free energy!");
    if (((F-Fold)/Fold > FENGYDEL) && (verbose == true)) // pos FE steps
      ostrm << '(' << (F-Fold) << ')' << flush;
    if (verbose == true)              // Notify iteration
      ostrm << '-' << flush;
    if ((i >= maxit) && (maxit > 0))  // Check max iter reached
      break;
  }
  while (abs(Fold-F)/Fold > CONVERGE);

  return i;
}


/* Split all of the mixtures.
 *  returns: the free energy of the best split
 *  mutable: qZ, the observation assignment probabilites
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
template <class W, class C> double split (
    const MatrixXd& X,                // Observations
    const MatrixXd& qZ,               // Assignment probs
    const libcluster::IGMC& igmc,     // The current I-GMC object
    const W& wdist,                   // Weight parameter distribution
    const vector<C>& cdists,          // Cluster parameter distributions
    MatrixXd& qZsplit,                // Assignment probs of best split
//    const bool sparse               // Do sparse updates
    const bool verbose,               // Verbose output
    ostream& ostrm                    // Stream to print notification
    )
{
  int K = qZ.cols(),
      Nj = qZ.rows();

  // Copy the weight and cluster parameter distributions for refinement
  W wdistref(wdist), wdistaug(wdist);
  vector<C> cdistref(2, cdists[0]), cdistaug(K+1, cdists[0]);

  // loop pre-allocations
  int M;
  ArrayXi mapidx;
  ArrayXb splitk;
  VectorXd Njk = qZ.colwise().sum();
  MatrixXd qZaug(Nj,K+1), qZref, Xk;
  double Ffree, Fbest = numeric_limits<double>::infinity(), Ftmp1, Ftmp2, Ftmp3;


  // Split each cluster and refine with VBEM. Record best split.
  for (int k=0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if ((cdists[k].getN() < 4) || (Njk(k) < 2))
      continue;

    // Make a copy of the observations with only relevant data points, p > 0.5
    partX(X, qZ.col(k), Xk, mapidx);
    M = Xk.rows();

    // Split the cluster
    splitk = cdists[k].splitobs(Xk);

    // Set up VBEM for refining split
    qZref = MatrixXd::Zero(M, 2);
    qZref.col(0) = (splitk == true).cast<double>(); // Initial qZ for split
    qZref.col(1) = (splitk == false).cast<double>();

    // Refine this split using VBEM
    libcluster::IGMC igmcsplt(igmc, k); // Copy suff. stats. of cluster k
    vbem<W, C>(Xk,qZref,igmcsplt,wdistref,cdistref,Ftmp1,Ftmp2,Ftmp3,SPLITITER);
    if (anyempty<C>(cdistref) == true)   // One cluster only
      continue;

    // Create new qZ for all data with split
    qZaug = augmentqZ(k, mapidx, (qZref.col(1).array() > 0.5), qZ);

    // Calculate free energy of this split with ALL data (and refine again)
    vbem<W, C>(X, qZaug, igmc, wdistaug, cdistaug, Ffree, Ftmp1, Ftmp2, 1);

    // Just check the clusters from the split merging back to one cluster
    if ((cdistaug[k].getN() <= 1) || (cdistaug[k+1].getN() <= 1))
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper (free energy)
    if (Ffree <= Fbest)
    {
      Fbest   = Ffree;
      qZsplit = qZaug;
    }
  }

  // Return best Free energy
  return Fbest;
}


/* Remove empty clusters from qZ, the I-GMC object, and the cluster parameter
 *  distribution vector.
 *  mutable: All arguments
 *  returns: The number of clusters removed
 */
template <class C> int remempty (
    MatrixXd& qZ,               // Assignment probs
    libcluster::IGMC& igmc,     // The current I-GMC object
    vector<C>& cdists           // Cluster parameter distributions
  )
{
  int K = cdists.size();

  if (K <= 1)
    return 0;

  // Find empty clusters if any, return if not
  ArrayXb keepidx = ArrayXb::Ones(K);

  for (int k = 0; k < K; ++k)
    if (cdists[k].getN() < 1)
      keepidx(k) = false;

  int keepcnt = keepidx.count();
  if (keepcnt == K)
    return 0;

  // Delete empty suff stats and reform qZ.
  MatrixXd qZnew(qZ.rows(), keepcnt);

  for (int k = 0, idx = 0; k < K; ++k)
  {
    if (keepidx(k) == true)
    {
      qZnew.col(idx) = qZ.col(k);
      ++idx;
    }
    else
    {
      cdists.erase(cdists.begin()+idx);
      if (idx <= igmc.getK())
        igmc.delSS(idx);
    }
  }

  qZ = qZnew;
  return (K - keepcnt); // return the number of removed Gaussians
}


//
// Public Functions
//

bool libcluster::learnIGMC (
  const MatrixXd& X,
  libcluster::IGMC& igmc,
  const bool verbose,
  ostream& ostrm
  )
{
  int K = (igmc.getK() < 1) ? 1 : igmc.getK();

  if (X.cols() != igmc.getD())
    throw invalid_argument("Dimensionality of X does not match I-GMC class!");

  // Re-create priors from the values stored in the I-GMC class
  GDirichlet wdist;
  vector<GaussWish> cdists(K, GaussWish(igmc.getcwidthp(), igmc.getcmeanp()));

  // Get an initial estimate of qZ from the I-GMC using classify if K > 1
  MatrixXd qZ, qZsplit;
  if (igmc.getK() <= 1)
    qZ = MatrixXd::Ones(X.rows(), 1);
  else
    libcluster::classifyIGMC(X, igmc, qZ);

  // Initialise free energy and other loop variables
  bool changedigmc = false;
  unsigned int it;
  double F, F_wj, F_xzj, Fsplt;

  // Start VBEM learning
  if (verbose == true)
    ostrm << "Learning incremental GMC..." << endl;

  while (true)
  {
    // Run "batch" VB for this group of data (runtime_error & invalid_argument)
    it = vbem<GDirichlet, GaussWish>(X, qZ, igmc, wdist, cdists, F, F_wj, F_xzj,
                                     -1, verbose, ostrm);

    if (it > 2)   // check if any significant VB iterations were performed
      changedigmc = true;

    // Remove any empty mixtures
    if (remempty<GaussWish>(qZ, igmc, cdists) > 0)
    {
      changedigmc = true;
      if ((verbose == true))
        ostrm << 'x' << flush;  // Notify removed some clusters
    }

    // Start looking for potential cluster splits on this group
    if (verbose == true)
      ostrm << '<' << flush;  // Notify start splitting
    Fsplt = split<GDirichlet, GaussWish>(X, qZ, igmc, wdist, cdists, qZsplit,
                                         verbose, ostrm);

    if (verbose == true)
      ostrm << '>' << endl;   // Notify end splitting

    // Choose either the split candidates, or finish!
    if ((Fsplt < F) && (abs(F-Fsplt)/F > CONVERGE))
    {
      cdists.push_back(cdists[0]); // copy new element from an existing
      qZ = qZsplit;
      changedigmc = true;
    }
    else
      break;  // Done!
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    if (changedigmc == true)
      ostrm << "I-GMC updated, Number of clusters = " << cdists.size() << endl;
    else
      ostrm << "I-GMC did not change significantly." << endl;
  }

  // Count the observations
  K = qZ.cols();
  double N = 0;
  for (int k=0; k < K; ++k)
    N += cdists[k].getN();

  // Create GMM
  vector<RowVectorXd> mu(K);
  vector<MatrixXd> sigma(K);
  vector<double> w(K);

  for (int k=0; k < K; ++k)
  {
    cdists[k].getmeancov(mu[k], sigma[k]);
    w[k] = cdists[k].getN()/N;
  }

  // Update I-GMC suff stats and GMM parameters
  igmc.update(X, qZ, w, mu, sigma, F_wj, F_xzj);

  return changedigmc;
}


RowVectorXd libcluster::classifyIGMC (
  const MatrixXd& X,
  const libcluster::IGMC& igmc,
  MatrixXd& qZ,
  const bool verbose,
  ostream& ostrm
  )
{
  int K = igmc.getK();

  if (X.cols() != igmc.getD())
    throw invalid_argument("Dimensionality of X does not match I-GMC class!");
  if (K == 0)
    throw invalid_argument("Please learn I-GMC first by calling learnIGMC()!");
  else if (K == 1)
  {
    if (verbose == true)
      ostrm << "Classifying using incremental GMC: One class.";
    qZ = MatrixXd::Ones(X.rows(), 1); // Trivial clustering result
    return RowVectorXd::Ones(1);
  }

  // Re-create priors and initialise posteriors from I-GMC from the values
  //  stored in the I-GMC class
  GDirichlet wdist;
  vector<GaussWish> cdists(K, GaussWish(igmc.getcwidthp(), igmc.getcmeanp()));
  for (int k = 0; k < K; ++k)
    cdists[k].update(igmc.getN_s(k), igmc.getx_s(k), igmc.getxx_s(k));

  // Get an initial estimate of qZ from the I-GMC GMM
  qZ = libcluster::classify(X, igmc);

  // Start VBEM Classification
  if (verbose == true)
    ostrm << "Classifying using I-GMC: ";

  // Initialise free energy and other loop variables
  double F = numeric_limits<double>::max(), Fold, F_wj, F_xzj, Ftheta;
  fenergy<GDirichlet, GaussWish>(wdist, cdists, F_wj, F_xzj);

  do
  {
    Fold = F;

    // VBM for this - only update the weight parameter distribution
    wdist.update(qZ.colwise().sum());

    // VBE for this group
    F_xzj = vbexpectation<GDirichlet, GaussWish>(X, qZ, wdist, cdists);

    // Calculate the free energy of the model
    fenergy<GDirichlet, GaussWish>(wdist, cdists, F_wj, F_xzj);
    igmc.calcF(F_wj, F_xzj); // Add in discount factors for Fpi and Fxz
    F = Ftheta + F_wj + F_xzj;

    if (F < 0)            // Check for bad free energy calculation
      throw runtime_error("Calculated a negative free energy!");
    if (verbose == true)  // Notify iteration
      ostrm << '-' << flush;
  }
  while (abs(Fold-F)/Fold > CONVERGE);

  // Print finished notification if verbose
  if (verbose == true)
    ostrm << " |" << endl;

  // Return this group's weights
  return wdist.getNk()/X.rows();
}
