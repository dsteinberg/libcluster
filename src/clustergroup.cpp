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

/* The Variational Bayes Maximisation step for the Grouped mixture models.
 *  mutable: wdists, the weight parameter distributions.
 *  mutable: cdists, model cluster parameter distributions.
 */
template <class W, class C> void vbmaximisation (
    const vector<MatrixXd>& X,  // Observations
    const vector<MatrixXd>& qZ, // Observations to model mixture assignments
    vector<W>& wdists,          // Weight parameter distributions
    vector<C>& cdists,          // Cluster parameter distributions
    const bool sparse           // Do sparse updates to groups
    )
{
  int K = cdists.size(),
      D = X[0].cols(),
      J = X.size() ;

  ArrayXd Njk;
  MatrixXd qZkXj;
  vector<double> Nk(K, 0);
  vector<RowVectorXd> x_s(K, RowVectorXd::Zero(D));
  vector<MatrixXd> xx_s(K, MatrixXd::Zero(D, D));

  // Update weight parameter distributions and get cluster suff. stats.
  for (int j = 0; j < J; ++j)
  {
    // Count obs. belonging to each cluster in this group, update weight dists.
    Njk = qZ[j].colwise().sum();
    wdists[j].update(Njk);

    // Sufficient statistics - can use 'sparsity' in these for speed
    for (int k = 0; k < K; ++k)
    {
      if ( (Njk(k) >= ZEROCUTOFF) || (sparse == false) )
      {
        qZkXj.noalias()   = qZ[j].col(k).asDiagonal() * X[j];
        Nk[k]             += Njk(k);
        x_s[k]            += qZkXj.colwise().sum();
        xx_s[k].noalias() += qZkXj.transpose() * X[j];
      }
    }
  }

  // Update the cluster parameter distributions
  for (int k = 0; k < K; ++k)
    cdists[k].update(Nk[k], x_s[k], xx_s[k]);
}


/* The Variational Bayes Expectation step for each group.
 *  mutable: Assignment probabilities, qZj
 *  returns: The complete-data (X,Z) free energy E[log p(X,Z)/q(Z)] for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class W, class C> double vbexpectationj (
    const MatrixXd& Xj,         // Observations in group J
    MatrixXd& qZj,              // Observations to group mixture assignments
    const W& wdistj,            // Group Weight parameter distribution
    const vector<C>& cdists,    // Cluster parameter distributions
    const bool sparse           // Do sparse updates to groups
    )
{
  int K = qZj.cols(),
      Nj = Xj.rows();

  ArrayXd E_logZ = wdistj.Emarginal();
  ArrayXb Kpop = (wdistj.getNk() >= ZEROCUTOFF);

  int nKpop = (sparse == true) ? Kpop.count() : K;
  MatrixXd logqZj(Nj, nKpop);

  // Find Expectations of log joint observation probs -- allow sparse evaluation
  for (int k = 0, sidx = 0; k < K; ++k)
  {
    if ( (Kpop(k) == true) || (sparse == false) )
    {
      logqZj.col(sidx) = E_logZ(k) + cdists[k].Eloglike(Xj).array();
      ++sidx;
    }
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZzj = logsumexp(logqZj);

  // Compute Responsabilities -- again allow sparse evaluation
  for (int k = 0, sidx = 0; k < K; ++k)
  {
    if ( (Kpop(k) == true) || (sparse == false) )
    {
      qZj.col(k) = ((logqZj.col(sidx) - logZzj).array().exp()).matrix();
      ++sidx;
    }
    else
      qZj.col(k) = VectorXd::Zero(Nj);
  }

  return -logZzj.sum();
}


/* Calculates the free energy lower bound for the model parameter distributions.
 *  returns: the free energy of the parameter distributions
 */
template <class W, class C> double fenergy (
    const vector<W>& wdists,    // Weight parameter distributions
    const vector<C>& cdists     // Cluster parameter distributions
    )
{
  int K = cdists.size(),
      J = wdists.size();

  double Fw = 0, Fc = 0;

  // Free energy of the weight parameter distributions
  for (int j = 0; j < J; ++j)
    Fw += wdists[j].fenergy();

  // Free energy of the cluster parameter distributions
  for (int k = 0; k < K; ++k)
    Fc += cdists[k].fenergy();

  return Fw + Fc;
}


/* Batch Varational Bayes EM for all group mixtures in the GMC.
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: wdists, all group weight parameter distributions.
 *  mutable: cdists, the model cluster parameter distributions.
 *  throws: invalid_argument rethrown from other functions or if cdists.size()
 *          does not match qZ[j].cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class C> double vbem (
    const vector<MatrixXd>& X,  // Observations
    vector<MatrixXd>& qZ,       // Observations to model mixture assignments
    vector<W>& wdists,          // Weight parameter distributions
    vector<C>& cdists,          // Cluster parameter distributions
    const int maxit = -1,       // Max 'meta' iterations (-1 = none, default)
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false, // Verbose output (default false)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    )
{
  int J = X.size();

  // Make sure cluster posterior vectors is right size
  if (cdists.size() != (unsigned) qZ[0].cols())
    throw invalid_argument("Wrong number of cluster parameter distributions!");

  int i = 0;
  double F = numeric_limits<double>::max(), Fold, Fxz;

  do
  {
    Fold = F;
    Fxz = 0;

    // VBM
    vbmaximisation<W, C>(X, qZ, wdists, cdists, sparse);

    // VBE
    for (int j = 0; j < J; ++j)
      Fxz += vbexpectationj<W, C>(X[j], qZ[j], wdists[j], cdists, sparse);

    // Calculate free energy of model
    F = fenergy<W, C>(wdists, cdists) + Fxz;

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > FENGYDEL)
      throw runtime_error("Free energy increase!");

    // Print iteration notification if verbose, check max iterations
    ++i;
    if (verbose == true)              // Notify iteration
      ostrm << '-' << flush;
    if ((i >= maxit) && (maxit > 0))  // Check max iter reached
      break;
  }
  while (abs(Fold-F)/Fold > CONVERGE);

  return F;
}


/* Split all mixtures, and return split that lowers model F the most.
 *  returns: The free energy of the best split, and the assignments in qZsplit.
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
template <class W, class C> double split (
    const vector<MatrixXd>& X,  // Observations
    const vector<MatrixXd>& qZ, // Probabilities qZ
    const vector<W>& wdists,    // Weight parameter distributions
    const vector<C>& cdists,    // Cluster parameter dists.
    vector<MatrixXd>& qZsplit,  // Probabilities qZ of split
    const bool sparse,          // Do sparse updates to groups
    const bool verbose,         // Verbose output
    ostream& ostrm              // Stream to print notification
    )
{
  int J = X.size(),
      K = cdists.size();

  // Pre allocate stuff for loops
  int scount, M, Mtot;
  double Fbest = numeric_limits<double>::infinity(), Fsp;
  ArrayXb splitk;
  vector<ArrayXi> mapidx(J, ArrayXi());
  vector<MatrixXd> qZref(J,MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd());

  // Copy the weight and cluster parameter distributions for refinement
  vector<W> wdistaug = wdists, wdistref = wdists;
  vector<C> cdistaug(K+1, cdists[0]), cdistref(2, cdists[0]);

  // Loop through each potential cluster and split it
  for (int k = 0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if (cdists[k].getN() < 4)
      continue;

    // Now split observations and qZ.
    scount = 0;
    Mtot   = 0;
    for (int j = 0; j < J; ++j)
    {
      // Make copy of the observations with only relevant data points, p > 0.5
      partX(X[j], qZ[j].col(k), Xk[j], mapidx[j]);
      M = Xk[j].rows();
      Mtot += M;

      splitk = cdists[k].splitobs(Xk[j]);

      // Set up VBEM for refining split
      qZref[j] = MatrixXd::Zero(M, 2);
      qZref[j].col(0) = (splitk == true).cast<double>(); // Init qZ for split
      qZref[j].col(1) = (splitk == false).cast<double>();

      // keep a track of number of splits
      scount += (splitk == true).count();
    }

    // Don't waste time with clusters that haven't been split sufficiently
    if ((scount < 2) | (scount > (Mtot-2)))
      continue;

    // Refine the split
    vbem<W,C>(Xk, qZref, wdistref, cdistref, SPLITITER, sparse);
    if (anyempty<C>(cdistref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    for (int j = 0; j < J; ++j)
      qZaug[j] = augmentqZ(k, mapidx[j], (qZref[j].col(1).array()>0.5), qZ[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    Fsp = vbem<W,C>(X, qZaug, wdistaug, cdistaug, 1, sparse);
    if (anyempty<C>(cdistaug) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper, store it's attributes
    if (Fsp < Fbest)
    {
      qZsplit = qZaug;
      Fbest   = Fsp;
    }
  }

  // Return free energy
  return Fbest;
}


/* The model selection algorithm for a grouped mixture model.
 *  returns: Free energy of the final model
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: wdists, the weight parameter distributions.
 *  mutable: cdists, model cluster parameter distributions.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if there is a negative free energy or if free energy
 *          increases.
 */
template <class W, class C> double modelselection (
    const vector<MatrixXd>& X,  // Observations
    vector<MatrixXd>& qZ,       // Observations to model mixture assignments
    vector<W>& wdists,          // Weight parameter distributions
    vector<C>& cdists,          // Cluster parameter distributions
    const bool sparse,          // Do sparse updates to groups
    const bool verbose,         // Verbose output
    ostream& ostrm              // Stream to print notification
    )
{
  int J = X.size();

  if (cdists.size() < 1)
    throw invalid_argument("Need to instantiate at least one cluster dist.");

  // Initialise qZ
  qZ.clear();
  for (int j = 0; j < J; ++j)
    qZ.push_back(MatrixXd::Ones(X[j].rows(), 1));

  // Initialise free energy and other loop variables
  vector<MatrixXd> qZsplit;
  double F = numeric_limits<double>::max(), Fsplit;

  while (true)
  {
    // VBEM for all groups (throws runtime_error & invalid_argument)
    F = vbem<W, C>(X, qZ, wdists, cdists, -1, sparse, verbose, ostrm);

    // Start cluster splitting
    if (verbose == true)
      ostrm << '<' << flush;    // Notify start splitting

    // Search for best split, return its free energy.
    Fsplit = split<W, C>(X, qZ, wdists, cdists, qZsplit, sparse, verbose,
                            ostrm);

    if (verbose == true)
      ostrm << '>' << endl;   // Notify end splitting

    // Choose either the split candidates, or finish!
    if ((Fsplit < F) && (abs(F-Fsplit)/F > CONVERGE))
    {
      cdists.push_back(cdists[0]); // copy new element from an existing
      qZ = qZsplit;
    }
    else
      break;  // Done!
  }

  int K = cdists.size();

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    ostrm << "Number of clusters = " << K << endl;
    ostrm << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//

double libcluster::learnGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<RowVectorXd>& w,
    libcluster::GMM& gmm,
    const bool sparse,
    const bool verbose,
    const double clustwidth,
    ostream& ostrm
    )
{
  int J = X.size();

  // Construct the priors
  vector<GDirichlet> wdists(J, GDirichlet());
  vector<GaussWish> cdists(1, GaussWish(clustwidth, mean(X), cov(X)));

  // Model selection and Variational Bayes learning
  if (verbose == true)
    ostrm << "Learning GMC..." << endl;

  double F = modelselection<GDirichlet, GaussWish>(X, qZ, wdists, cdists,
                                                   sparse, verbose, ostrm);

  // Create GMM from model parameters, and get group weights
  gmm = makeGMM(cdists);

  w.resize(J);
  for (int j = 0; j < J; ++j)
    w[j] = wdists[j].getNk()/X[j].rows();
  return F;
}


double libcluster::learnSGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<RowVectorXd>& w,
    libcluster::GMM& gmm,
    const bool sparse,
    const bool verbose,
    const double clustwidth,
    ostream& ostrm
    )
{
  int J = X.size();

  // Construct the priors
  vector<Dirichlet> wdists(J, Dirichlet());
  vector<GaussWish> cdists(1, GaussWish(clustwidth, mean(X), cov(X)));

  // Model selection and Variational Bayes learning
  if (verbose == true)
    ostrm << "Learning Symmetric GMC..." << endl;

  double F = modelselection<Dirichlet, GaussWish>(X, qZ, wdists, cdists,
                                                   sparse, verbose, ostrm);

  // Create GMM from model parameters, and get group weights
  gmm = makeGMM(cdists);

  w.resize(J);
  for (int j = 0; j < J; ++j)
    w[j] = wdists[j].getNk()/X[j].rows();

  return F;
}
