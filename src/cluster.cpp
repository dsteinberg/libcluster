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
// Private Functions
//

/* The Variational Bayes Maximisation step.
 *  mutable: wdist, the weight parameter distribution.
 *  mutable: cdists, model cluster parameter distributions.
 *  throws: invalid_argument if iW is not PSD.
 */
template <class W, class C> void vbmaximisation (
    const MatrixXd& X,          // Observations
    const MatrixXd& qZ,         // Assignment probabilities
    W& wdist,                   // Mixture weight param. dist.
    vector<C>& cdists           // Cluster parameter dist.
    )
{
  int K = qZ.cols(),
      N = X.rows(),
      D = X.cols();

  // Update weight parameter distributions
  ArrayXd Nk = qZ.colwise().sum();
  wdist.update(Nk);

  // Update cluster parameter distributions
  RowVectorXd x_s(D);
  MatrixXd xx_s(D, D), qZkX(N, D);
  for (int k = 0; k < K; ++k)
  {
    // Create the sufficient statistics of the mixed data
    qZkX.noalias() = qZ.col(k).asDiagonal() * X;
    x_s            = qZkX.colwise().sum();
    xx_s.noalias() = qZkX.transpose() * X;

    // Update the Gaussian Wishart Parameters using the sufficient statistics
    cdists[k].update(Nk(k), x_s, xx_s);
  }
}


/* The Variational Bayes Expectation step.
 *  mutable: Assignment probabilities, qZ
 *  returns: The complete-data (X,Z) free energy E[log p(X,Z)/q(Z)].
 *  throws: invalid_argument from other functions
 */
template <class W, class C> double vbexpectation (
    const MatrixXd& X,        // Observations
    MatrixXd& qZ,             // Assignment probabilities
    const W& wdist,           // Mixture weight param. dist.
    const vector<C>& cdists   // Cluster parameter dist.
    )
{
  int K = qZ.cols(),
      N = X.rows();

  // Calculate expected log weights
  const ArrayXd E_logZ = wdist.Emarginal();

  // Expectations of cluster log likelihood
  MatrixXd logqZ(N, K);
  for (int k = 0; k < K; ++k)
    logqZ.col(k) = E_logZ(k) + cdists[k].Eloglike(X).array();

  // Log normalisation constant of log observation likelihoods
  VectorXd logZz = logsumexp(logqZ);

  // Compute Responsabilities/complete data log likelihood
  qZ = ((logqZ.colwise() - logZz).array().exp()).matrix();

  return -logZz.sum();
}


/* Calculates the free energy lower bound for the model parameter distributions.
 *  returns: the free energy of the parameter distributions
 */
template <class W, class C> double fenergy (
    const W& wdist,           // Mixture weight param. dist.
    const vector<C>& cdists   // Cluster parameter dist.
    )
{
  // Calculate the cluster parameter free energy terms
  double Fc = 0;
  for (unsigned int k = 0; k < cdists.size(); ++k)
    Fc += cdists[k].fenergy();

  // Return sum of the parameter free energy
  return wdist.fenergy() + Fc;
}


/* The VBEM algorithm for a mixture model. Defaults to no maximum iterations.
 *  returns: Free energy
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: wdist, the weight parameter distribution.
 *  mutable: cdists, model cluster parameter distributions.
 *  throws: invalid_argument from other functions or if cdists.size() does not
 *          match qZ.cols().
 *  throws: runtime_error if there is a negative free energy or if free energy
 *          increases
 */
template <class W, class C> double vbem (
    const MatrixXd& X,          // Observations
    MatrixXd& qZ,               // Assignment probabilities
    W& wdist,                   // Mixture weight param. dist.
    vector<C>& cdists,          // Cluster parameter dist.
    const int maxit = -1,       // Max iterations
    const bool verbose = false, // Verbose output
    ostream& ostrm = cout       // Stream to print notifications
    )
{
  unsigned int K = qZ.cols();

  // Make sure cluster posterior vectors is right size
  if (cdists.size() != K)
    throw invalid_argument("Wrong number of cluster parameter distributions!");

  // VBEM loop
  int i = 0;
  double F = numeric_limits<double>::max(), Fold = F, Fxz;

  do
  {
    // Maximisation (VBM)
    vbmaximisation<W, C>(X, qZ, wdist, cdists);

    // Expectations & Responsabilities (VBE)
    Fxz = vbexpectation<W, C>(X, qZ, wdist, cdists);

    // Free energy
    Fold = F;
    F = Fxz + fenergy<W, C>(wdist, cdists);

    // Check for bad steps
    if ((F-Fold)/abs(Fold) > FENGYDEL)
      throw runtime_error("Free energy increase!");

    // Print iteration notification if verbose, check max iterations
    ++i;
    if (verbose == true)
      ostrm << '-' << flush; // Iter notification
    if ((i >= maxit) && (maxit > 0))
      break;                 // Check max iter reached
  }
  while ((Fold-F)/Fold > CONVERGE);

  // Return free energy
  return F;
}


/* Split all of the mixtures.
 *  returns: the free energy of the best split and its assignments, qZsplit
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
template <class W, class C> double split (
    const MatrixXd& X,      // Observations
    const MatrixXd& qZ,     // Assignment probabilities
    const W& wdist,         // Mixture weight param. dist.
    const vector<C>& cdists,// Cluster parameter dist.
    MatrixXd& qZsplit,      // Assignment probabilities of model with best split
    const bool verbose,     // Verbose output
    ostream& ostrm          // Stream to print notification
    )
{
  int K = qZ.cols(),
      N = qZ.rows();

  // Copy the weight and cluster parameter distributions for refinement
  W wdistref(wdist), wdistaug(wdist);
  vector<C> cdistref(2, cdists[0]), cdistaug(K+1, cdists[0]);

  // loop pre-allocations
  int M, scount = 0;
  ArrayXi mapidx;
  ArrayXb splitk;
  MatrixXd qZaug(N,K+1), qZref, Xk;
  double Ffree, Fbest = numeric_limits<double>::infinity();

  // Split each cluster and refine with VBEM. Record best split.
  for (int k=0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if ( (wdist.getNk())[k] < 4 )
      continue;

    // Make a copy of the observations with only relevant data points, p > 0.5
    partX(X, qZ.col(k), Xk, mapidx);
    M = Xk.rows();

    // Split the cluster
    splitk = cdists[k].splitobs(Xk);

    // Don't waste time with clusters that haven't been split sufficiently
    scount = (splitk == true).count();
    if ((scount < 2) || (scount > (M-2)))
      continue;

    // Set up VBEM for refining split
    qZref = MatrixXd::Zero(M, 2);
    qZref.col(0) = (splitk == true).cast<double>(); // Initial qZ for split
    qZref.col(1) = (splitk == false).cast<double>();

    // Refine this split using VBEM
    vbem<W, C>(Xk, qZref, wdistref, cdistref, SPLITITER);
    if ( (wdistref.getNk() <= 1).any() == true )   // One cluster only
      continue;

    // Create new qZ for all data with split
    qZaug = augmentqZ(k, mapidx, (qZref.col(1).array() > 0.5), qZ);

    // Calculate free energy of this split with ALL data (and refine again)
    Ffree = vbem<W, C>(X, qZaug, wdistaug, cdistaug, 1);
    if ( (wdistaug.getNk() <= 1).any() == true )   // One cluster only
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


/* The model selection algorithm for a mixture model.
 *  returns: Free energy of the final model
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: wdist, the weight parameter distribution.
 *  mutable: cdists, model cluster parameter distributions.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if there is a negative free energy or if free energy
 *          increases.
 */
template <class W, class C> double modelselection (
    const MatrixXd& X,
    MatrixXd& qZ,
    W& wdist,
    vector<C>& cdists,
    const bool verbose,
    ostream& ostrm
    )
{
  if (cdists.size() < 1)
    throw invalid_argument("Need to instantiate at least one cluster dist.");

  // Make initial qZ a column vector of all ones of appropriate length
  qZ = MatrixXd::Ones(X.rows(), 1);

  double F, Fsplit;
  MatrixXd qZsplit;

  // Model Selection loop
  while (true)
  {
    // run VBEM (throws runtime_error & invalid_argument)
    F = vbem<W, C>(X, qZ, wdist, cdists, -1, verbose, ostrm);

    // Split and refine mixtures (throws runtime_error & invalid_argument)
    if (verbose == true)
      ostrm << '<' << flush; // Notify start splitting

    qZsplit = qZ; // Copy last qZ assignments, still want to keep last good qZ
    Fsplit = split<W, C>(X, qZ, wdist, cdists, qZsplit, verbose, ostrm);

    if (verbose == true)
      ostrm << '>' << endl;  // Notify end splitting

    // Choose either the split candidates, or finish!
    if((Fsplit < F) && (abs(F-Fsplit)/F > CONVERGE))
    {
      cdists.push_back(cdists[0]); // copy new element from an existing
      qZ = qZsplit;
    }
    else
      break;  // Done!
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    ostrm << "Number of clusters = " << qZ.cols() << endl;
    ostrm << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//

double libcluster::learnVDP (
    const MatrixXd& X,
    MatrixXd& qZ,
    libcluster::GMM& gmm,
    const bool verbose,
    const double clustwidth,
    ostream& ostrm
    )
{
  // Make parameter distributions and create priors
  StickBreak wdist;
  vector<GaussWish> cdists(1, GaussWish(clustwidth, mean(X), cov(X)));

  if (verbose == true)
    ostrm << "Learning VDP..." << endl; // Print start

  // Perform model learning and selection
  double F = modelselection<StickBreak, GaussWish>(X, qZ, wdist, cdists,
                                                   verbose, ostrm);

  // Create GMM
  gmm = makeGMM(cdists);

  // Return final Free energy and qZ
  return F;
}


double libcluster::learnGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    libcluster::GMM& gmm,
    const bool verbose,
    const double clustwidth,
    ostream& ostrm
    )
{

  // Make parameter distributions and create priors
  Dirichlet wdist;
  vector<GaussWish> cdists(1, GaussWish(clustwidth, mean(X), cov(X)));

  if (verbose == true)
    ostrm << "Learning Bayesian GMM..." << endl; // Print start

  // Perform model learning and selection
  double F = modelselection<Dirichlet, GaussWish>(X, qZ, wdist, cdists, verbose,
                                                  ostrm);

  // Create GMM
  gmm = makeGMM(cdists);

  // Return final Free energy and qZ
  return F;
}
