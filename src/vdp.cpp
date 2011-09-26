// TODO:
//  - catch exceptions when construction priors.
//  - There is a lot of qZ copying in the learnvdp and split functions. Can I
//    get rid of some of it?

#include <limits>
#include "libcluster.h"
#include "probutils.h"
#include "vbcommon.h"

//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace vbcommon;


//
// Prototypes
//

/* The VBEM algorithm for the VDP. Defaults to no maximum iterations.
 *  returns: Free energy
 *  mutable: assignment vector qZ
 *  mutable: vector<SBposterior> posterior. Does not need to be instantiated.
 *  mutable: vector<GWposterior> posterior. Does not need to be instantiated.
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy or if free energy
 *          increases
 */
double vbem (
    const MatrixXd& X,          // Observations
    MatrixXd& qZ,               // Assignment probabilities (mutable)
    const SBprior& sprior,      // Prior hyperparameter values
    const GWprior& gprior,      // Prior hyperparameter values
    vector<SBposterior>& spost, // Posterior Hyperparameter values (mutable)
    vector<GWposterior>& gpost, // Posterior Hyperparameter values (mutable)
    const int maxit = -1,       // Max iterations (-1 = none, default)
    const bool verbose = false, // Verbose output (default off = 0)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    );


/* The Variational Bayes Maximisation step.
 *  mutable: vectors of posterior variational parameters
 *  returns: An array of the descending size order of the clusters
 *  throws: invalid_argument if iW is not PSD.
 */
ArrayXi vbmaximisation (
    const MatrixXd& X,          // Observations
    const MatrixXd& qZ,         // Assignment probabilities
    const SBprior& sprior,      // Prior hyperparameter values
    const GWprior& gprior,      // Prior hyperparameter value
    vector<SBposterior>& spost, // Posterior Hyperparameter values (mutable)
    vector<GWposterior>& gpost  // Posterior Hyperparameter values (mutable)
    );


/* The Variational Bayes Expectation step.
 *  mutable: vector of Stick-breaking posterior variational parameters
 *  mutable: Assignment probabilities, qZ
 *  returns: negative sum of the log normalisation constant, sum_{n}(p(X)).
 *           This is also the free energy of the observations, Fz.
 *  throws: invalid_argument rethrown from other functions
 */
double vbexpectation (
    const MatrixXd& X,          // Observations
    MatrixXd& qZ,               // Assignment probabilities (mutable)
    vector<SBposterior>& spost, // Posterior Hyperparameter values (mutable)
    const vector<GWposterior>& gpost, // Posterior Hyperparameter values
    const ArrayXi& order        // Descending size order of the clusters
    );


/* Calculates the free energy lower bound.
 *  returns: the free energy of the parameter distributions
 */
double fenergy (
    const SBprior& sprior,            // Prior hyperparameter values
    const GWprior& gprior,            // Prior hyperparameter value
    const vector<SBposterior>& spost, // Posterior Hyperparameter values
    const vector<GWposterior>& gpost  // Posterior Hyperparameter values
    );


/* Split all of the mixtures.
 *  returns: the free energy of the best split
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
double split (
    const MatrixXd& X,                // Observations
    const MatrixXd& qZ,               // Assignment probs
    const SBprior& sprior,            // Prior hyperparameter values
    const GWprior& gprior,            // Prior hyperparameter value
    const vector<GWposterior>& gpost, // Posterior Hyperparameter values
    MatrixXd& qZsplit,                // Assignment probs of best split
    const bool verbose = false,       // Verbose output (default off = 0)
    ostream& ostrm = cout             // Stream to print notification
    );


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
  int N = X.rows();

  // Initialise repeatedly used variables
  double F, Fsplit;
  vector<SBposterior> spost;
  vector<GWposterior> gpost;

  // Make initial qZ a column vector of all ones of appropriate length
  qZ = MatrixXd::Ones(N, 1);
  MatrixXd qZsplit;

  // Construct the priors
  SBprior sprior;  // Uninformed Stick-breaking prior and expectations
  GWprior gprior(clustwidth, cov(X), mean(X)); // Informed G-W prior

  if (verbose == true)
    ostrm << "Learning VDP..." << endl; // Print start

  // Main loop
  while (true)
  {
    // run VBEM
    try
      { F = vbem(X, qZ, sprior, gprior, spost, gpost, -1, verbose, ostrm); }
    catch (...)
      { throw; }             // runtime_error & invalid_argument

    if (verbose == true)
      ostrm << '<' << flush; // Notify start splitting

    // Split and refine mixtures
    qZsplit = qZ; // Copy last qZ assignments, still want to keep last good qZ
    try
      { Fsplit = split(X, qZ, sprior, gprior, gpost, qZsplit, verbose, ostrm); }
    catch (...)
      { throw; }             // runtime_error & invalid_argument

    if (verbose == true)
      ostrm << '>' << endl;  // Notify end splitting

    // Choose either the split candidates, or finish!
    if((Fsplit < F) && (abs(F-Fsplit)/F > CONVERGE))
      qZ = qZsplit;
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

  // Create GMM
  vector<RowVectorXd> mu;
  vector<MatrixXd> sigma;
  vector<double> w;

  for (unsigned int k=0; k < gpost.size(); ++k)
  {
    mu.push_back(gpost[k].getm());
    sigma.push_back(gpost[k].getiW()/gpost[k].getnu());
    w.push_back(gpost[k].getNk()/N);
  }

  GMM retgmm(mu, sigma, w);
  gmm = retgmm;

  // Return final Free energy and qZ
  return F;
}


//
// Private Functions
//

double vbem (
    const MatrixXd& X,
    MatrixXd& qZ,
    const SBprior& sprior,
    const GWprior& gprior,
    vector<SBposterior>& spost,
    vector<GWposterior>& gpost,
    const int maxit,
    const bool verbose,
    ostream& ostrm
    )
{
  unsigned int K = qZ.cols();

  // Make sure posterior vectors are right size
  if (spost.size() != K)
    spost.resize(K, SBposterior());
  if (gpost.size() != K)
    gpost.resize(K, GWposterior());

  // VBEM loop
  int i = 0;
  double F = numeric_limits<double>::max(), Fold = F, Fz;
  RowVectorXi order;

  do
  {
    try
    {
      // Maximisation (VBM)
      order = vbmaximisation(X, qZ, sprior, gprior, spost, gpost);

      // Expectations & Responsabilities (VBE)
      Fz = vbexpectation(X, qZ, spost, gpost, order);

      // Free energy
      Fold = F;
      F = Fz + fenergy(sprior, gprior, spost, gpost);
    }
    catch (...)
      { throw; } // invalid_argument, runtime_error

    // Check for bad steps
    if (F < 0)
      throw runtime_error("Calculated a negative free energy!");
    if ((F-Fold)/Fold > FENGYDEL)
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


ArrayXi vbmaximisation (
    const MatrixXd& X,
    const MatrixXd& qZ,
    const SBprior& sprior,
    const GWprior& gprior,
    vector<SBposterior>& spost,
    vector<GWposterior>& gpost
    )
{
  int K = qZ.cols(),
      N = X.rows(),
      D = X.cols();

  // Create order independent posterior parameters
  double Nk;
  RowVectorXd xksum(D);
  MatrixXd Rksum(D, D), qZkX(N, D);
  vector<pair<int,double> > ordvec(K, pair<int,double>());  // order index

  for (int k = 0; k < K; ++k)
  {
    // Create the sufficient statistics of the mixed data
    qZkX  = qZ.col(k).asDiagonal() * X;
    Nk    = qZ.col(k).sum();
    xksum = qZkX.colwise().sum();
    Rksum = qZkX.transpose() * X;

    // Update the Gaussian Wishart Parameters using the sufficient statistics
    gpost[k].update(Nk, xksum, Rksum, gprior);

    // Record position and order of classes
    ordvec[k].first  = k;
    ordvec[k].second = Nk; // Cluster observation count
  }

  // Find cluster size order
  sort(ordvec.begin(), ordvec.end(), paircomp);

  // Update order dependent posterior parameters alpha2 (and alpha1 too)
  ArrayXi order = ArrayXi::Zero(K);
  double Nkcumsum = 0;

  for (int k = 0; k < K; ++k)
  {
    order(k) = ordvec[k].first;           // Create row order vector
    Nkcumsum += ordvec[k].second;         // Accumulate cluster size sum
    spost[order(k)].update(ordvec[k].second, (N-Nkcumsum), sprior);
  }

  return order;
}


double vbexpectation (
    const MatrixXd& X,
    MatrixXd& qZ,
    vector<SBposterior>& spost,
    const vector<GWposterior>& gpost,
    const ArrayXi& order
    )
{
  int K = qZ.cols(),
      N = X.rows();

  // Pre-allocate stuff for calculation of the expectations
  int k;
  double cumE_lognv = 0;
  VectorXd E_logX(N);
  MatrixXd logqZ(N, K);

  // Do everything in descending size order since it is easier on memory
  for (int idx=0; idx < K; ++idx)
  {
    k = order(idx); // Get the ordered index

    // Expectations of log stick lengths (we store E_logZ in the post struct)
    cumE_lognv += spost[k].Eloglike(cumE_lognv);

    // Expectations of log mixture likelihoods
    E_logX = gpost[k].Eloglike(X);

    // Expectations of log joint observation probs
    logqZ.col(k) = spost[k].getE_logZ() + E_logX.array();
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZzj = logsumexp(logqZ);

  // Compute Responsabilities
  qZ = ((logqZ.colwise() - logZzj).array().exp()).matrix();

  return -logZzj.sum();
}


double fenergy (
    const SBprior& sprior,            // Prior hyperparameter values
    const GWprior& gprior,            // Prior hyperparameter value
    const vector<SBposterior>& spost, // Posterior Hyperparameter values
    const vector<GWposterior>& gpost  // Posterior Hyperparameter values
    )
{
  int K = gpost.size();

  // Calculate the parameter free energy terms
  double Fv = 0, Fn = 0;

  // Free energy dependent on prior and posterior terms
  for (int k=0; k < K; ++k)
  {
    Fv += spost[k].fnrgpost(sprior);    // Stick breaking params
    Fn += gpost[k].fnrgpost(gprior);    // Cluster/Gaussian params
  }

  // Add in Free energy terms only dependent on priors.
  Fv += (K*sprior.fnrgprior());
  Fn += (K*gprior.fnrgprior());

  return Fv + Fn;
}


double split (
    const MatrixXd& X,
    const MatrixXd& qZ,
    const SBprior& sprior,
    const GWprior& gprior,
    const vector<GWposterior>& gpost,
    MatrixXd& qZsplit,
    const bool verbose,
    ostream& ostrm
    )
{
  int K = qZ.cols(),
      N = qZ.rows(),
      D = X.cols();

  // loop pre-allocations
  int M, scount = 0;
  ArrayXi mapidx;
  ArrayXb splitk;
  VectorXd eigvec(D); // Eigenvec for PC split
  MatrixXd qZk(N,K+1), qZr, Xk;
  double Ffree, Fbest = numeric_limits<double>::infinity();
  vector<SBposterior> spostsplit(2, SBposterior()),
                      spostfree(K+1, SBposterior());
  vector<GWposterior> gpostsplit(2, GWposterior()),
                      gpostfree(K+1, GWposterior());

  // Split each cluster perpendicular to P.C. and refine with VBEM. Find best
  //  split too.
  for (int k=0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if (gpost[k].getNk() < 4)
      continue;

    // Make a copy of the observations with only relevant data points, p > 0.5
    partX(X, qZ.col(k), Xk, mapidx);
    M = Xk.rows();

    // Find the principle component using the power method, 'split'
    //  observations assignments, qZ, perpendicular to it.
    eigpower(gpost[k].getrefiW(), eigvec);
    splitk = (((Xk.rowwise() - gpost[k].getrefm())  // PC project and split
             * eigvec.asDiagonal()).array().rowwise().sum()) >= 0;

    // Don't waste time with clusters that haven't been split sufficiently
    scount = (splitk == true).count();
    if ((scount < 2) || (scount > (M-2)))
      continue;

    // Set up VBEM for refining split
    qZr = MatrixXd::Zero(M, 2);
    qZr.col(0) = (splitk == true).cast<double>(); // Initial qZ for split
    qZr.col(1) = (splitk == false).cast<double>();

    // Refine this split using VBEM
    try
      { vbem(Xk, qZr, sprior, gprior, spostsplit, gpostsplit, SPLITITER); }
    catch (invalid_argument e)
      { throw invalid_argument(string("Refining split: ").append(e.what())); }
    catch (runtime_error e)
      { throw runtime_error(string("Refining split: ").append(e.what())); }

    if (checkempty<GWposterior>(gpostsplit) == true)   // One cluster only
      continue;

    // Create new qZ for all data with split
    qZk = augmentqZ(k, mapidx, (qZr.col(1).array() > 0.5), qZ);

    // Calculate free energy of this split with ALL data (and refine again)
    try
      { Ffree = vbem(X, qZk, sprior, gprior, spostfree, gpostfree, 1); }
    catch (invalid_argument e)
      { throw invalid_argument(string("Split FE: ").append(e.what())); }
    catch (runtime_error e)
      { throw runtime_error(string("Split FE: ").append(e.what())); }

    if (checkempty<GWposterior>(gpostfree) == true)  // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper (free energy)
    if (Ffree <= Fbest)
    {
      Fbest  = Ffree;
      qZsplit = qZk;
    }
  }

  // Return best Free energy
  return Fbest;
}
