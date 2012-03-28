// TODO:
//  - Get rid of the copying in the learnVDP and learnGMM functions.
//  - Some copying still in split()

#include <limits>
#include "libcluster.h"
#include "probutils.h"
#include "distributions.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace distributions;


//
// Some local 'symbolic' constants
//

const int    SPLITITER   = 20;          // Max number of iter. for split VBEM
const double CONVERGE    = 1.0e-5;      // Convergence threshold
const double FENGYDEL    = CONVERGE/10; // Allowance for +ve F.E. steps
const double ZEROCUTOFF  = 0.1;         // Obs. count cut off sparse updates


//
// Private Helper structures and functions
//

/* Triplet that contains the information for choosing a good cluster split
 *  ordering.
 */
struct SplitOrder
{
  int k;      // Cluster number/index
  int tally;  // Number of times a cluster has failed to split
  double Fk;  // The clusters approximate free energy contribution
};


/* Compares two SplitOrder triplets and returns which is more optimal to split.
 *  Precendence is given to less split fail tally, and then to more free energy
 *  contribution.
 */
bool inline splitcomp (const SplitOrder& i, const SplitOrder& j)
{
  if (i.tally == j.tally)       // If the tally is the same, use the greater Fk
    return i.Fk > j.Fk;
  else if (i.tally < j.tally)   // Otherwise prefer the lower tally
    return true;
  else
    return false;
}


/* Find the indices of the ones and zeros in a binary array.
 *
 *  mutable: indtrue the indices of the true values in the array "expression"
 *  mutable: indfalse the indices of the false values in the array "expression"
 */
void arrfind (const ArrayXb& expression, ArrayXi& indtrue, ArrayXi& indfalse)
{
  int N = expression.size(),
      M = expression.count();
  indtrue  = ArrayXi::Zero(M);
  indfalse = ArrayXi::Zero(N-M);

  for (int n = 0, m = 0, l = 0; n < N; ++n)
    expression(n) ? indtrue(m++) = n : indfalse(l++) = n;
}


/* Partition the observations, X according to a logical array.
 *
 *  mutable: Xk, MxD matrix of observations that have a correspoding 1 in Xpart.
 *  returns: an Mx1 array of the locations of Xk in X.
 */
ArrayXi partX (
    const MatrixXd& X,    // NxD matrix of observations.
    const ArrayXb& Xpart, // Nx1 indicator vector to partition X by.
    MatrixXd& Xk          // MxD matrix of obs. beloning to new partition
    )
{
  int M = Xpart.count();

  ArrayXi pidx, npidx;
  arrfind(Xpart, pidx, npidx);

  Xk = MatrixXd::Zero(M, X.cols());
  for (int m=0; m < M; ++m)           // index copy X to Xk
    Xk.row(m) = X.row(pidx(m));

  return pidx;
}


/* Augment the assignment matrix, qZ with the split cluster entry.
 *
 * The new cluster assignments are put in the K+1 th column in the return matrix
 *  returns: The new observation assignments, [Nx(K+1)].
 *  throws: std::invalid_argument if map.size() != Zsplit.size().
 */
MatrixXd  augmentqZ (
    const double k,        // The cluster to split (i.e. which column of qZ)
    const ArrayXi& map,    // Mapping from array of partitioned obs to qZ
    const ArrayXb& Zsplit, // Boolean array of assignments.
    const MatrixXd& qZ     // [NxK] observation assignment probability matrix.
    )
{
  int K = qZ.cols(),
      N = qZ.rows(),
      S = Zsplit.count();

  if (Zsplit.size() != map.size())
    throw invalid_argument("map and split must be the same size!");

  // Create new qZ for all data with split
  MatrixXd qZaug = qZ;    // Copy the existing qZ into the new
  qZaug.conservativeResize(Eigen::NoChange, K+1);
  qZaug.col(K) = VectorXd::Zero(N);

  ArrayXi sidx, nsidx;
  arrfind(Zsplit, sidx, nsidx);

  // Copy split cluster assignments (augment qZ effectively)
  for (int s = 0; s < S; ++s)
  {
    qZaug(map(sidx(s)), K) = qZ(map(sidx(s)), k); // Add new cluster onto end
    qZaug(map(sidx(s)), k) = 0;
  }

  return qZaug;
}


/* Check if any sufficient statistics are empty.
 *
 *  returns: True if any of the sufficient statistics are empty
 */
bool anyempty (const libcluster::SuffStat& SS)
{
  int K = SS.getK();

  for (int k = 0; k < K; ++k)
    if (SS.getN_k(k) <= 1)
      return true;

  return false;
}


//
// Private Algorithm Functions
//

/* Update the group and model sufficient statistics based on assignments qZj.
 *
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 */
template <class C> void updateSSj (
    const MatrixXd& Xj,         // Observations in group j
    const MatrixXd& qZj,        // Observations to group mixture assignments
    libcluster::SuffStat& SSj,  // Sufficient stats of group j
    libcluster::SuffStat& SS,   // Sufficient stats of whole model
    const bool sparse           // Do sparse updates to groups
    )
{
  unsigned int K = qZj.cols();

  #pragma omp critical
  SS.subSS(SSj);                      // get rid of old group SS contribution

  ArrayXd Njk = qZj.colwise().sum();  // count obs. in this group
  ArrayXi Kful = ArrayXi::Zero(1),    // Initialise and set K = 1 defaults
          Kemp = ArrayXi::Zero(0);
  MatrixXd SS1, SS2;                  // Suff. Stats

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((Njk >= ZEROCUTOFF), Kful, Kemp);

  int nKful = Kful.size(), nKemp = Kemp.size();

  // Sufficient statistics - with observations
  for (int k = 0; k < nKful; ++k)
  {
    C::makeSS(qZj.col(Kful(k)), Xj, SS1, SS2);
    SSj.setSS(Kful(k), Njk(Kful(k)), SS1, SS2);
  }

  // Sufficient statistics - without observations
  Array4i dimSS = C::dimSS(Xj);
  for (int k = 0; k < nKemp; ++k)
    SSj.setSS(Kemp(k), 0, MatrixXd::Zero(dimSS(0), dimSS(1)),
              MatrixXd::Zero(dimSS(2), dimSS(3)));

  #pragma omp critical
  SS.addSS(SSj); // Add new group SS contribution
}


/* The Variational Bayes Maximisation step for the group mixture weights.
 *
 *  mutable: wdists, the weight parameter distributions.
 */
template <class W> void vbmaximisationj (
    const libcluster::SuffStat& SSj, // Sufficient stats of each group
    W& wdistj                        // Weight parameter distributions
    )
{
  unsigned int K = SSj.getK();

  ArrayXd Njk = ArrayXd::Zero(K);

  // Get cluster counts for group
  for (unsigned int k = 0; k < K; ++k)
    Njk(k) = SSj.getN_k(k);

  // Update the weight parameter distribution
  wdistj.update(Njk);
}


/* The Variational Bayes Expectation step for each group.
 *
 *  mutable: Group assignment probabilities, qZj
 *  returns: The complete-data (X,Z) free energy E[log p(X,Z)/q(Z)] for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class W, class C> double vbexpectationj (
    const MatrixXd& Xj,         // Observations in group J
    const W& wdistj,            // Group Weight parameter distribution
    const vector<C>& cdists,    // Cluster parameter distributions
    MatrixXd& qZj,              // Observations to group mixture assignments
    const bool sparse           // Do sparse updates to groups
    )
{
  int K  = cdists.size(),
      Nj = Xj.rows();

  // Get log marginal weight likelihoods
  ArrayXd E_logZ = wdistj.Eloglike();

  // Initialise and set K = 1 defaults for cluster counts
  ArrayXi Kful = ArrayXi::Zero(1), Kemp = ArrayXi::Zero(0);

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((wdistj.getNk() >= ZEROCUTOFF), Kful, Kemp);

  int nKful = Kful.size(),
      nKemp = Kemp.size();

  // Find Expectations of log joint observation probs -- allow sparse evaluation
  MatrixXd logqZj(Nj, nKful);

  for (int k = 0; k < nKful; ++k)
    logqZj.col(k) = E_logZ(Kful(k)) + cdists[Kful(k)].Eloglike(Xj).array();

  // Log normalisation constant of log observation likelihoods
  VectorXd logZzj = logsumexp(logqZj);

  // Make sure qZ is the right size, this is a nop if it is
  qZj.resize(Nj, K);

  // Compute Responsibilities -- again allow sparse evaluation
  for (int k = 0; k < nKful; ++k)
    qZj.col(Kful(k)) = ((logqZj.col(k) - logZzj).array().exp()).matrix();

  // Empty Cluster Responsabilities
  for (int k = 0; k < nKemp; ++k)
    qZj.col(Kemp(k)) = VectorXd::Zero(Nj);

  return -logZzj.sum();
}


/* Calculates the free energy lower bound for the model parameter distributions.
 *
 *  returns: the free energy of the model
 *  mutable: the group sufficient statistics - for the purposes of recording F
 *  mutable: the model sufficient statistics - for the purposes of recording F
 */
template <class W, class C> double fenergy (
    const vector<W>& wdists,            // Weight parameter distributions
    const vector<C>& cdists,            // Cluster parameter distributions
    const vector<double>& Fxz,          // Free energy from data log-likelihood
    vector<libcluster::SuffStat>& SSj,  // Group sufficient statistics
    libcluster::SuffStat& SS            // Model Sufficient statistics
    )
{
  int K = cdists.size(),
      J = wdists.size();

  // Free energy of the weight parameter distributions
  for (int j = 0; j < J; ++j)
  {
    SS.subF(SSj[j]);  // Remove old groups F contribution
    SSj[j].setF(wdists[j].fenergy() + Fxz[j]);
    SS.addF(SSj[j]);  // Add in the new groups F contribution
  }

  // Free energy of the cluster parameter distributionsreturn
  double Fc = 0;
  for (int k = 0; k < K; ++k)
    Fc += cdists[k].fenergy();

  return Fc + SS.getF();
}


/* Batch Variational Bayes EM for all group mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument rethrown from other functions or if cdists.size()
 *          does not match qZ[j].cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class C> double vbem (
    const vector<MatrixXd>& X,  // Observations
    vector<MatrixXd>& qZ,       // Observations to model mixture assignments
    vector<libcluster::SuffStat>& SSj, // Sufficient stats of each group
    libcluster::SuffStat& SS,   // Sufficient stats of whole model
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false, // Verbose output (default false)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    )
{
  int J = X.size(),
      K = qZ[0].cols();

  // Construct the parameters
  vector<W> wdists(X.size(), W());
  vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

  double F = numeric_limits<double>::max(), Fold;
  vector<double> Fxz(J);
  int i = 0;

  do
  {
    Fold = F;

    // Update Suff Stats and VBM for weights
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < J; ++j)
    {
      updateSSj<C>(X[j], qZ[j], SSj[j], SS, sparse);
      vbmaximisationj<W>(SSj[j], wdists[j]);
    }

    // VBM for clusters
    #pragma omp parallel for schedule(dynamic)
    for (int k=0; k < K; ++k)
      cdists[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));

    // VBE
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < J; ++j)
      Fxz[j] = vbexpectationj<W,C>(X[j], wdists[j], cdists, qZ[j], sparse);

    // Calculate free energy of model
    F = fenergy<W,C>(wdists, cdists, Fxz, SSj, SS);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > FENGYDEL)
      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      ostrm << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > CONVERGE) && ( (i++ < maxit) || (maxit < 0) ) );

  return F;
}


/*  Find a mixture split that lowers model free energy, or return false.
 *    An attempt is made at looking for good, untried, split candidates first,
 *    as soon as a split canditate is found that lowers model F, it is returned.
 *    This may not be the "best" split, but it is certainly faster.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    mutable tally is a tally time a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
template <class W, class C> bool split (
    const vector<MatrixXd>& X,               // Observations
    const vector<libcluster::SuffStat>& SSj, // Sufficient stats of groups
    const libcluster::SuffStat& SS,          // Sufficient stats
    const double F,                          // Current model free energy
    vector<MatrixXd>& qZ,                    // Probabilities qZ
    vector<int>& tally,                      // Count of unsuccessful splits
    const bool sparse,                       // Do sparse updates to groups
    const bool verbose,                      // Verbose output
    ostream& ostrm                           // Stream to print notification
    )
{
  unsigned int J = X.size(),
               K = SS.getK();

  // Split order chooser and cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<SplitOrder> ord(K);
  vector<C> csplit(K, C(SS.getprior(), X[0].cols()));

  // Get cluster parameters and their free energy
  #pragma omp parallel for schedule(dynamic)
  for (unsigned int k = 0; k < K; ++k)
  {
    csplit[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = csplit[k].fenergy();
  }

  // Get cluster likelihoods
  #pragma omp parallel for schedule(dynamic)
  for (unsigned int j = 0; j < J; ++j)
  {
    // Get cluster weights
    W wsplit;
    wsplit.update(qZ[j].colwise().sum());
    ArrayXd logpi = wsplit.Eloglike();

    // Add in cluster log-likelihood, weighted by responsability
    for (unsigned int k = 0; k < K; ++k)
    {
      double L = logpi(k) + qZ[j].col(k).dot(csplit[k].Eloglike(X[j]));

      #pragma omp atomic
      ord[k].Fk -= L;
    }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), splitcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector<ArrayXi> mapidx(J, ArrayXi());
  vector<MatrixXd> qZref(J,MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd());

  // Loop through each potential cluster in order and split it
  for (vector<SplitOrder>::iterator i = ord.begin(); i < ord.end(); ++i)
  {
    int k = i->k;

    // Don't waste time with clusters that can't really be split min (2:2)
    if (SS.getN_k(k) < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+ : Mtot, scount)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      mapidx[j] = partX(X[j], (qZ[j].col(k).array()>0.5), Xk[j]);  // Copy :-(
      Mtot += Xk[j].rows();

      // Initial cluster split
      ArrayXb splitk = csplit[k].splitobs(Xk[j]);
      qZref[j] = MatrixXd::Zero(Xk[j].rows(), 2);
      qZref[j].col(0) = (splitk == true).cast<double>();  // Init qZ for split
      qZref[j].col(1) = (splitk == false).cast<double>();

      // keep a track of number of splits
      scount += splitk.count();
    }

    // Don't waste time with clusters that haven't been split sufficiently
    if ( (scount < 2) || (scount > (Mtot-2)) )
      continue;

    // Refine the split
    libcluster::SuffStat SSref(SS.getprior());
    vector<libcluster::SuffStat> SSgref(J, libcluster::SuffStat(SS.getprior()));
    vbem<W,C>(Xk, qZref, SSgref, SSref, SPLITITER, sparse);
    if (anyempty(SSref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    #pragma omp parallel for schedule(dynamic)
    for (unsigned int j = 0; j < J; ++j)
      qZaug[j] = augmentqZ(k, mapidx[j], (qZref[j].col(1).array()>0.5), qZ[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    libcluster::SuffStat SSaug = SS;                              // Copy :-(
    vector<libcluster::SuffStat> SSj_aug = SSj;                   // Copy :-(
    double Fsplit = vbem<W,C>(X, qZaug, SSj_aug, SSaug, 1, sparse);
    if (anyempty(SSaug) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > CONVERGE) )
    {
      qZ = qZaug;
      tally[k] = 0;
      return true;
    }
    else
      ++tally[k]; // increase this cluster's unsuccessful split tally
  }

  // Failed to find splits
  return false;
}


/*  Bootstrap the clustering models. I.e. make sure we have reasonable starting
 *    assignments, qZ depending on whether there are previous suff. stats.
 *
 *    mutable: group sufficient stats if they need to be initialised
 *    mutable: qZ for a good starting position for VBEM
 *    throws: invalid_argument If there are some group sufficient stats. but
 *            they do not correspond to the number of groups in X.
 */
template <class W, class C> void bootstrap (
  const vector<MatrixXd>& X,          // Observations
  const libcluster::SuffStat& SS,     // Model sufficient stats
  vector<libcluster::SuffStat>& SSj,  // Group sufficient stats
  vector<MatrixXd>& qZ                // Obs. to model mixture assignments
  )
{
  unsigned int K = (SS.getK() < 1) ? 1 : SS.getK(),
               J = X.size();

  // Create or check the group sufficient stats
  if (SSj.size() == 0)
    SSj.resize(X.size(), libcluster::SuffStat(SS.getprior()));
  else if (SSj.size() != J)
    throw invalid_argument("SSj does not have the same no. of groups as X!");

  qZ.resize(J);

  // Create a good starting qZ depending on if we have an old model or not
  if (SS.getK() > 0)  // We have previous sufficient stats., make up clusters
  {
    ArrayXd Nk  = ArrayXd::Zero(K);
    ArrayXd Njk = Nk;

    // Construct the priors
    vector<W> wdists(X.size(), W());
    vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

    // Create cluster params from old suff stats
    for (unsigned int k = 0; k < K; ++k)
    {
      cdists[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));
      Nk(k) = SS.getN_k(k);
    }

    // Create weights and preliminary labels
    for (unsigned int j = 0; j < J; ++j)
    {
     if (SSj[j].getK() > 0)  // Use old group suff. stats. if we have some
      {
        for (unsigned int k = 0; k < K; ++k)
          Njk(k) = SSj[j].getN_k(k);

        wdists[j].update(Njk);
      }
      else                  // otherwise just use the model weights
        wdists[j].update(Nk);

      vbexpectationj<W,C>(X[j], wdists[j], cdists, qZ[j], false);
    }
  }
  else  // This is an entirely new model to learn, make one cluster
  {
    for (unsigned int j = 0; j < J; ++j)
      qZ[j] = MatrixXd::Ones(X[j].rows(), 1);
  }
}


/* The model selection algorithm for a grouped mixture model.
 *
 *  returns: Free energy of the final model
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if free energy increases.
 */
template <class W, class C> double learnmodel (
    const vector<MatrixXd>& X,  // Observations
    vector<MatrixXd>& qZ,       // Observations to model mixture assignments
    vector<libcluster::SuffStat>& SSgroups, // Sufficient stats of groups
    libcluster::SuffStat& SS,   // Sufficient stats
    const bool sparse,          // Do sparse updates to groups
    const bool verbose,         // Verbose output
    ostream& ostrm              // Stream to print notification
    )
{  
  // Bootstrap qZ, cdists and wdists
  bootstrap<W,C>(X, SS, SSgroups, qZ);

  // Initialise free energy and other loop variables
  bool   issplit = true;
  double F = numeric_limits<double>::max();
  vector<int> tally;

  // Main loop
  while (issplit == true)
  {
    // VBEM for all groups (throws runtime_error & invalid_argument)
    F = vbem<W,C>(X, qZ, SSgroups, SS, -1, sparse, verbose, ostrm);

    // Start cluster splitting
    if (verbose == true)
      ostrm << '<' << flush;  // Notify start splitting

    // Search for best split, augment qZ if found one
    issplit = split<W,C>(X, SSgroups, SS, F, qZ, tally, sparse, verbose, ostrm);

    if (verbose == true)
      ostrm << '>' << endl;   // Notify end splitting
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    ostrm << "Number of clusters = " << SS.getK() << endl;
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
    SuffStat& SS,
    const bool diagcov,
    const bool verbose,
    ostream& ostrm
    )
{
  if (verbose == true)
    ostrm << "Learning VDP..." << endl; // Print start

  // Make temporary vectors of data to use with learnmodel()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F;
  if (diagcov == false)
    F = learnmodel<StickBreak, GaussWish>(vecX, vecqZ, SSgroup, SS, false,
                                         verbose, ostrm);
  else
    F = learnmodel<StickBreak, NormGamma>(vecX, vecqZ, SSgroup, SS, false,
                                         verbose, ostrm);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                        // copy :-(
  return F;
}


double libcluster::learnGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    SuffStat& SS,
    const bool diagcov,
    const bool verbose,
    ostream& ostrm
    )
{
  if (verbose == true)
    ostrm << "Learning Bayesian GMM..." << endl; // Print start

  // Make temporary vectors of data to use with learnmodel()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<libcluster::SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F;
  if (diagcov == false)
    F = learnmodel<Dirichlet, GaussWish>(vecX, vecqZ, SSgroup, SS, false,
                                         verbose, ostrm);
  else
    F = learnmodel<Dirichlet, NormGamma>(vecX, vecqZ, SSgroup, SS, false,
                                         verbose, ostrm);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copy :-(
  return F;
}


double libcluster::learnGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse,
    const bool diagcov,
    const bool verbose,
    ostream& ostrm
    )
{
  // Model selection and Variational Bayes learning
  if (verbose == true)
    ostrm << "Learning GMC..." << endl;

  double F;
  if (diagcov == false)
    F = learnmodel<GDirichlet, GaussWish>(X, qZ, SSgroups, SS, sparse, verbose,
                                          ostrm);
  else
    F = learnmodel<GDirichlet, NormGamma>(X, qZ, SSgroups, SS, sparse, verbose,
                                               ostrm);

  return F;
}


double libcluster::learnSGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse,
    const bool diagcov,
    const bool verbose,
    ostream& ostrm
    )
{
  // Model selection and Variational Bayes learning
  if (verbose == true)
    ostrm << "Learning Symmetric GMC..." << endl;

  double F;
  if (diagcov == false)
    F = learnmodel<Dirichlet, GaussWish>(X, qZ, SSgroups, SS, sparse, verbose,
                                          ostrm);
  else
    F = learnmodel<Dirichlet, NormGamma>(X, qZ, SSgroups, SS, sparse, verbose,
                                               ostrm);

  return F;
}
