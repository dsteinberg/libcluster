// TODO -- maybe
//  - Handle GWprior() constructor throwing errors!
//  - There is a lot of qZ copying in the splitall function. Can I get rid of
//    some of it?

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
// Local Symbolic Constants
//

const double ZEROCUTOFF = 0.1;      // Obs. count, Nk cut off for sparse updates


//
// Private Types, Globals and Structures
//

// Model Gaussian-Wishart Parameter structure with group-sufficient statistics
class GWsuffpost : public GWposterior
{
public:

  // Constructor for the posterior parameters requires number of groups, J,
  //  and dimensionality of data, D, to initialise sufficient stats vectors.
  GWsuffpost (int J, int D)
  {
    Njk = ArrayXd::Zero(J);
    xjk = ArrayXXd::Zero(J, D);
    Rjk.resize(J, MatrixXd::Zero(D, D));
  }

  // Sufficient statistics from the groups for this model cluster
  ArrayXd Njk;          // sufficient stat for group cluster counts
  ArrayXXd xjk;         // sufficient stat for means, m (J rows)
  vector<MatrixXd> Rjk; // sufficient stat for covariances, iW
};


//
// Prototypes
//

/* Batch Varational Bayes EM for all group mixtures in the GMC.
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: spost, all group posterior parameters.
 *  mutable: gpost, the model gpost parameters.
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
double vbembat (
    const vector<MatrixXd>& X,  // Observations
    vector<MatrixXd>& qZ,       // Observations to model mixture assignments
    const SBprior& sprior,      // Model SB priors
    const GWprior& gprior,      // Model GW priors
    vector< vector<SBposterior> >& spost, // all posterior group parameters
    vector<GWsuffpost>& gpost,  // posterior model parameters
    const int maxit = -1,       // Max 'meta' iterations (-1 = none, default)
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false, // Verbose output (default false)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    );

/* The Variational Bayes Maximisation step for the group weights in the GMC.
 * This function also creates the group-wise sufficient statistics for
 * calculating the model parameters.
 *  returns: An array of the descending size order of the group clusters
 *  mutable: spostj, the group posterior parameters.
 *  mutable: gpost, model posterior parameters (their sufficient stats).
 */
ArrayXi vbmaximisationj (
    const int groupj,            // The group to update the suff. stats. for
    const MatrixXd& Xj,          // Observations in group j
    const MatrixXd& qZj,         // Observations to model mixture assignments
    const SBprior& sprior,       // Group SB priors
    vector<SBposterior>& spostj, // posterior group SB parameters
    vector<GWsuffpost>& gpost,   // posterior model GW parameters
    const bool sparse = false    // Do sparse updates to groups (default false)
    );

/* The Variational Bayes Maximisation step for the model cluster in the GMC.
 *  mutable: gpost, the model posterior parameters.
 *  throws: invalid_argument if iW is not PSD.
 *  NOTE: vbmaximisationj() should be run before this to update the suff.
 *        stats. so this can then update the model parameters
 */
void vbmaximisationk (
    const GWprior& gprior,     // Model GW priors
    vector<GWsuffpost>& gpost, // posterior GW model parameters
    const bool sparse = false  // Do sparse updates (default false)
    );

/* The Variational Bayes Expectation step for each group.
 *  mutable: spostj, vector of group posterior variational parameters
 *  mutable: Assignment probabilities, qZj
 *  returns: negative sum of the log normalisation constant, sum_{n}(p(Xj)).
 *           This is also the free energy of the observations, Fzj.
 *  throws: invalid_argument rethrown from other functions.
 *  NOTE: Only use this function's return value to calculate the free energy
 *        of the entire model when batch learning. Fz needs to be calculated
 *        for the most recent value of the parameters, otherwise it will be
 *        invalid!
 */
double vbexpectationj (
    const int groupj,           // The group to update the expectations for
    const MatrixXd& Xj,         // Observations in group J
    MatrixXd& qZj,              // Observations to group mixture assignments
    vector<SBposterior>& spostj,     // Posterior SB group parameters
    const vector<GWsuffpost>& gpost, // Posterior GW model parameters
    const ArrayXi& order,       // Descending size order of the group clusters
    const bool sparse = false   // Do sparse updates to groups (default false)
    );

/* Calculates the free energy for terms that factor over groups, j.
 *  returns: the free energy of the parameter distribution, Vj, over j.
 */
double fenergySB (
    const SBprior& sprior,             // Group SB priors
    const vector<SBposterior>& spostj, // Posterior SB group parameters
    const ArrayXi& order               // Cluster weight order.
    );

/* Calculates the free energy for terms that factor over model clusters, k.
 *  returns: the free energy of the parameter distributions over k.
 */
double fenergyGW (
    const GWprior& gprior,          // Model GW priors
    const vector<GWsuffpost>& gpost // Posterior GW model parameters
    );

/* Split all mixtures, and return split that lowers model F the most.
 *  returns: The free energy of the best split (total GMC model free energy).
 *  mutable: gpost, the posterior hyperparameters of the best split
 *  mutable: qZ, the observation assignment probabilites, returns best split
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
double splitall (
    const vector<MatrixXd>& X,  // Observations
    const vector<MatrixXd>& qZ, // Probabilities qZ
    const SBprior& sprior,      // Prior SB group hyperparameter values
    const GWprior& gprior,      // Prior GW model hyperparameter values
    const vector<GWsuffpost>& gpost, // Posterior model hyperparameters
    vector<GWsuffpost>& gpostsplit,  // Posterior model hyperparameters of split
    vector<MatrixXd>& qZsplit,  // Probabilities qZ of split
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false, // Verbose output (default off = 0)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    );


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
  int J = X.size(),
      D = X[0].cols();

  // Test X for consistency
  for (int j = 0; j < J; ++j)
    if (X[j].cols() != D)
      throw invalid_argument("X dimensions are inconsistent between groups!");

  // Construct the prior
  GWprior gprior(clustwidth, cov(X), mean(X));
  SBprior sprior;

  // Initialise qZ, and initialise the model posterior sufficient stats.
  int N = 0;
  qZ.clear();
  vector<MatrixXd> qZsplit;
  vector<GWsuffpost> gpost(1, GWsuffpost(J, D)), gpostsplit;
  vector< vector<SBposterior> > spost; // Group parameters

  for (int j = 0; j < J; ++j)
  {
    // Initialise qZ
    qZ.push_back(MatrixXd::Ones(X[j].rows(), 1));

    // Initialise the sufficient statistics and group parameter vectors
    spost.push_back(vector<SBposterior>(1, SBposterior()));

    // Get number of observations
    N += X[j].rows();
  }

  // Initialise free energy and other loop variables
  double F = numeric_limits<double>::max(), Fsplit;

  // Start VBEM learning
  if (verbose == true)
    ostrm << "Learning GMC..." << endl;

  while (true)
  {
    // VBEM for all groups
    try
      { F = vbembat(X,qZ,sprior,gprior,spost,gpost,-1,sparse,verbose,ostrm); }
    catch(...)
      { throw; }

    // Start cluster splitting
    if (verbose == true)
      ostrm << '<' << flush;    // Notify start splitting

    // Search for best split, return its free energy.
    try
    {
      Fsplit = splitall(X, qZ, sprior, gprior, gpost, gpostsplit, qZsplit,
                        sparse, verbose, ostrm);
    }
    catch(...)
      { throw; }

    if (verbose == true)
      ostrm << '>' << endl;   // Notify end splitting

    // Choose either the split candidates, or finish!
    if ((Fsplit < F) && (abs(F-Fsplit)/F > CONVERGE))
    {
      qZ    = qZsplit;
      gpost = gpostsplit;

      for (int j = 0; j < J; ++j)      // Augment group parameters.
        spost[j].push_back(SBposterior());
    }
    else
      break;  // Done!
  }

  int K = gpost.size();

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    ostrm << "Number of clusters = " << K << endl;
    ostrm << "Free energy = " << F << endl;
  }

  // Create GMM from model parameters, and get group weights
  vector<RowVectorXd> mu;
  vector<MatrixXd> sigma;
  vector<double> wK;
  w.clear();
  w.resize(J, RowVectorXd::Zero(K));

  for (int k = 0; k < K; ++k)
  {
    mu.push_back(gpost[k].getm());
    sigma.push_back(gpost[k].getiW()/gpost[k].getnu());
    wK.push_back(gpost[k].getNk()/N);

    for (int j = 0; j < J; ++j)
      w[j](k) = exp(spost[j][k].getE_logZ()); // exp{ E[log p(z_j=k)] }
  }

  GMM retgmm(mu, sigma, wK);
  gmm = retgmm;
  return F;
}


//
// Private Functions
//

double vbembat (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    const SBprior& sprior,
    const GWprior& gprior,
    vector< vector<SBposterior> >& spost,
    vector<GWsuffpost>& gpost,
    const int maxit,
    const bool sparse,
    const bool verbose,
    ostream& ostrm
    )
{
  int J = X.size(),
      K = gpost.size();

  int i = 0;
  double F = numeric_limits<double>::max(), Fold, Fpi, Fx;
  vector<ArrayXi> order(J, ArrayXi::Zero(K));

  do
  {
    Fold = F;
    Fpi = 0;
    Fx = 0;

    try
    {
      // VBM Groups
      for (int j = 0; j < J; ++j)
        order[j] = vbmaximisationj(j,X[j],qZ[j],sprior,spost[j],gpost,sparse);

      // VBM Model
      vbmaximisationk(gprior, gpost, sparse);

      // VBE
      for (int j = 0; j < J; ++j)
        Fx += vbexpectationj(j, X[j], qZ[j], spost[j], gpost, order[j], sparse);
    }
    catch (...)
      { throw; }

    // Calculate free energy of model
    for (int j = 0; j < J; ++j)
      Fpi += fenergySB(sprior, spost[j], order[j]);
    F = Fpi + fenergyGW(gprior, gpost) + Fx;

    ++i;
    if (F < 0)                        // Check for bad free energy calculation
      throw runtime_error("Calculated a negative free energy!");
    if ((F-Fold)/Fold > FENGYDEL)     // Check bad free energy step
      throw runtime_error("Free energy increase!");
    if (verbose == true)              // Notify iteration
      ostrm << '-' << flush;
    if ((i >= maxit) && (maxit > 0))   // Check max iter reached
      break;
  }
  while (abs(Fold-F)/Fold > CONVERGE);

  return F;
}


ArrayXi vbmaximisationj (
    const int groupj,
    const MatrixXd& Xj,
    const MatrixXd& qZj,
    const SBprior& sprior,
    vector<SBposterior>& spostj,
    vector<GWsuffpost>& gpost,
    const bool sparse
    )
{
  int K  = qZj.cols(),
      D  = Xj.cols(),
      Nj = Xj.rows();

  // Vector for sorting the group clusters in size order
  vector<pair<int,double> > ordvec(K, pair<int,double>());  // order index

  // Update all of the group suff. stats. and non-order dependent params in
  //  each cluster
  MatrixXd qZkXj(Nj, D);

  for (int k = 0; k < K; ++k)
  {
    // Sufficient statistics - can use 'sparsity' in these for speed
    gpost[k].Njk(groupj) = qZj.col(k).sum();
    if ( (gpost[k].Njk(groupj) >= ZEROCUTOFF) || (sparse == false) )
    {
      qZkXj = qZj.col(k).asDiagonal() * Xj;
      gpost[k].xjk.row(groupj) = qZkXj.colwise().sum();
      gpost[k].Rjk[groupj]     = qZkXj.transpose() * Xj;
    }
    else // probably don't need to do this - but it's good practice...
    {
      gpost[k].xjk.row(groupj) = RowVectorXd::Zero(D);
      gpost[k].Rjk[groupj]     = MatrixXd::Zero(D, D);
    }

    // Record cluster size and position
    ordvec[k].first  = k;
    ordvec[k].second = gpost[k].Njk(groupj);
  }

  // Sort the clusters in size order (descending)
  sort(ordvec.begin(), ordvec.end(), paircomp);

  // Now update the order dependent terms
  ArrayXi order = ArrayXi::Zero(K);
  double Njkcumsum = 0;

  for (int k = 0; k < K; ++k)
  {
    order(k) = ordvec[k].first;         // Create row order vector
    Njkcumsum += ordvec[k].second;      // Accumulate cluster size sum
    spostj[order(k)].update(ordvec[k].second, (Nj-Njkcumsum), sprior);
  }

  return order;
}


void vbmaximisationk (
    const GWprior& gprior,
    vector<GWsuffpost>& gpost,
    const bool sparse
    )
{
  int K = gpost.size(),
      J = gpost[0].Njk.size(),
      D = gpost[0].xjk.cols();

  // Initialise various statistics and the order vector
  RowVectorXd xjksum(D);
  MatrixXd Rjksum(D,D);

  // Update the unordered gpost model parameters
  for (int k = 0; k < K; ++k)
  {
    // Create relevant cluster statistics from sufficient statistics
    double Nk = gpost[k].Njk.sum();
    xjksum = RowVectorXd::Zero(D);
    Rjksum = MatrixXd::Zero(D, D);

    for (int j = 0; j < J; ++j)
    {
      if ( (gpost[k].Njk(j) >= ZEROCUTOFF) || (sparse == false) ) //allow sparse
      {
        xjksum += gpost[k].xjk.row(j).matrix();
        Rjksum += gpost[k].Rjk[j];
      }
    }

    // Update the Gaussian Wishart Parameters using the sufficient statistics
    gpost[k].update(Nk, xjksum, Rjksum, gprior);
  }
}


double vbexpectationj (
    const int groupj,
    const MatrixXd& Xj,
    MatrixXd& qZj,
    vector<SBposterior>& spostj,
    const vector<GWsuffpost>& gpost,
    const ArrayXi& order,
    const bool sparse
    )
{
  int K = qZj.cols(),
      Nj = Xj.rows();

  bool truncate;
  int k;
  double cumE_lognvj = 0;
  ArrayXb Kpop(K);

  // Do following in descending size order (also need to calc all of these)
  for (int idx = 0; idx < K; ++idx)
  {
    k = order(idx); // Get the ordered index

    // Find out which clusters have observations in this group
    Kpop(k) = gpost[k].Njk(groupj) >= ZEROCUTOFF;

    // Expectations of log stick lengths (we store E_logZj in the post struct)
    truncate = (idx == (K-1)) ? true : false; // truncate the GD here?
    cumE_lognvj += spostj[k].Eloglike(cumE_lognvj, truncate);
  }

  int nKpop = (sparse == true) ? Kpop.count() : K;
  MatrixXd logqZj(Nj, nKpop);

  // Find Expectations of log joint observation probs -- allow sparse evaluation
  for (int k = 0, sidx = 0; k < K; ++k)
  {
    if ( (Kpop(k) == true) || (sparse == false) )
    {
      logqZj.col(sidx) = spostj[k].getE_logZ() + gpost[k].Eloglike(Xj).array();
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


double fenergySB (
    const SBprior& sprior,
    const vector<SBposterior>& spostj,
    const ArrayXi& order
    )
{
  int K = spostj.size();

  // If there is only one cluster, there are no SB parameters for a GD
  if (K == 1)
    return 0;

  // Free energy dependent on prior and posterior terms, for K-1 SB params,
  //  leaving out the last K because the SB weight is just 1.
  double Fpi = 0;
  for (int k = 0; k < (K-1); ++k)
    Fpi += spostj[order(k)].fnrgpost(sprior);

  // Add in Free energy terms only dependent on priors.
  Fpi += ((K-1)*sprior.fnrgprior()); // Add in constant prior bits
  return Fpi;
}


double fenergyGW (
    const GWprior& gprior,
    const vector<GWsuffpost>& gpost
    )
{
  int K = gpost.size();

  // Free energy dependent on prior and posterior terms
  double Fn = 0;
  for (int k=0; k < K; ++k)
    Fn += gpost[k].fnrgpost(gprior);

  // Add in Free energy terms only dependent on priors.
  Fn += (K*gprior.fnrgprior()); // Add in constant prior bits
  return Fn;
}


double splitall (
    const vector<MatrixXd>& X,
    const vector<MatrixXd>& qZ,
    const SBprior& sprior,
    const GWprior& gprior,
    const vector<GWsuffpost>& gpost,
    vector<GWsuffpost>& gpostsplit,
    vector<MatrixXd>& qZsplit,
    const bool sparse,
    const bool verbose,
    ostream& ostrm
    )
{
  int J = X.size(),
      K = gpost.size(),
      D = X[0].cols();

  // Pre allocate stuff for loops
  int scount, M, Mtot;
  double Fbest = numeric_limits<double>::infinity(), Fsp;
  VectorXd eigvec(D); // Eigenvec for PC split
  ArrayXb splitk;
  vector<ArrayXi>     mapidx(J, ArrayXi());
  vector<MatrixXd>    qZref(J, MatrixXd()), qZsp(J, MatrixXd()),
                      Xk(J, MatrixXd());
  vector<GWsuffpost>  gpostcp(K+1, GWsuffpost(J,D)),
                      gpostref(2, GWsuffpost(J,D));

  // Preallocate SB posterior group vectors
  vector <vector<SBposterior> > spostcp, spostref;
  for (int j = 0; j < J; ++j)
  {
    spostcp.push_back(vector<SBposterior>(K+1, SBposterior()));
    spostref.push_back(vector<SBposterior>(2, SBposterior()));
  }

  // Loop through each potential cluster and split it
  for (int k = 0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if (gpost[k].getNk() < 4)
      continue;

    // Find the principal eigenvector using the power method
    eigpower(gpost[k].getrefiW(), eigvec);

    // Now split observations and qZ perpendicular to principal eigenvector.
    scount = 0;
    Mtot   = 0;
    for (int j = 0; j < J; ++j)
    {

      // Make copy of the observations with only relevant data points, p > 0.5
      partX(X[j], qZ[j].col(k), Xk[j], mapidx[j]);
      M = Xk[j].rows();
      Mtot += M;

      splitk = (((Xk[j].rowwise() - gpost[k].getrefm()) // Split perp. to PC
               * eigvec.asDiagonal()).array().rowwise().sum()) >= 0;

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
    try
      { vbembat(Xk,qZref,sprior,gprior,spostref,gpostref,SPLITITER,sparse); }
    catch(...)
      { throw; }

    if (checkempty<GWsuffpost>(gpostref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    for (int j = 0; j < J; ++j)
      qZsp[j] = augmentqZ(k, mapidx[j], (qZref[j].col(1).array() > 0.5), qZ[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    try
      { Fsp = vbembat(X, qZsp, sprior, gprior, spostcp, gpostcp, 1, sparse); }
    catch(...)
      { throw; }

    if (checkempty<GWsuffpost>(gpostcp) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper, store it's attributes
    if (Fsp < Fbest)
    {
      gpostsplit = gpostcp;
      qZsplit    = qZsp;
      Fbest     = Fsp;
    }
  }

  // Return free energy
  return Fbest;
}
