// TODO:
//  - Get rid of the copying in the learnVDP and learnGMM functions.
//  - Some copying still in split()

#include <limits>
#include "libcluster.h"
#include "probutils.h"
#include "distributions.h"
#include "comutils.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace probutils;
using namespace distributions;
using namespace comutils;


//
// Private Algorithm Functions
//

/* Update the group and model sufficient statistics based on assignments qZj.
 *
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 */
template <class C> void updateSS (
    const MatrixXd& Xj,         // Observations in group j
    const MatrixXd& qZj,        // Observations to group mixture assignments
    libcluster::SuffStat& SSj,  // Sufficient stats of group j
    libcluster::SuffStat& SS,   // Sufficient stats of whole model
    const bool sparse           // Do sparse updates to groups
    )
{
  const unsigned int K = qZj.cols();

  #pragma omp critical
  SS.subSS(SSj);                      // get rid of old group SS contribution

  const ArrayXd Njk = qZj.colwise().sum();  // count obs. in this group
  ArrayXi Kful = ArrayXi::Zero(1),          // Initialise and set K = 1 defaults
          Kemp = ArrayXi::Zero(0);
  MatrixXd SS1, SS2;                        // Suff. Stats

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((Njk >= libcluster::ZEROCUTOFF), Kful, Kemp);

  const int nKful = Kful.size(),
            nKemp = Kemp.size();

  // Sufficient statistics - with observations
  for (int k = 0; k < nKful; ++k)
  {
    C::makeSS(qZj.col(Kful(k)), Xj, SS1, SS2);
    SSj.setSS(Kful(k), Njk(Kful(k)), SS1, SS2);
  }

  // Sufficient statistics - without observations
  const pair<Array2i, Array2i> dimSS = C::dimSS(Xj);
  for (int k = 0; k < nKemp; ++k)
    SSj.setSS(Kemp(k), 0,
              MatrixXd::Zero(dimSS.first(0), dimSS.first(1)),
              MatrixXd::Zero(dimSS.second(0), dimSS.second(1)));

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
  const unsigned int K = SSj.getK();

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
template <class W, class C> double vbexpectation (
    const MatrixXd& Xj,         // Observations in group J
    const W& wdistj,            // Group Weight parameter distribution
    const vector<C>& cdists,    // Cluster parameter distributions
    MatrixXd& qZj,              // Observations to group mixture assignments
    const bool sparse           // Do sparse updates to groups
    )
{
  const int K  = cdists.size(),
            Nj = Xj.rows();

  // Get log marginal weight likelihoods
  const ArrayXd E_logZ = wdistj.Eloglike();

  // Initialise and set K = 1 defaults for cluster counts
  ArrayXi Kful = ArrayXi::Zero(1), Kemp = ArrayXi::Zero(0);

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((wdistj.getNk() >= libcluster::ZEROCUTOFF), Kful, Kemp);

  const int nKful = Kful.size(),
            nKemp = Kemp.size();

  // Find Expectations of log joint observation probs -- allow sparse evaluation
  MatrixXd logqZj(Nj, nKful);

  for (int k = 0; k < nKful; ++k)
    logqZj.col(k) = E_logZ(Kful(k)) + cdists[Kful(k)].Eloglike(Xj).array();

  // Log normalisation constant of log observation likelihoods
  const VectorXd logZzj = logsumexp(logqZj);

  // Make sure qZ is the right size, this is a nop if it is
  qZj.resize(Nj, K);

  // Normalise and Compute Responsibilities -- again allow sparse evaluation
  for (int k = 0; k < nKful; ++k)
    qZj.col(Kful(k)) = ((logqZj.col(k) - logZzj).array().exp()).matrix();

  // Empty Cluster Responsabilities
  for (int k = 0; k < nKemp; ++k)
    qZj.col(Kemp(k)).setZero();

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
  const int K = cdists.size(),
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


/* Variational Bayes EM for all group mixtures.
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
    const bool verbose = false  // Verbose output (default false)
    )
{
  const int J = X.size(),
            K = qZ[0].cols();

  // Construct the parameters
  vector<W> wdists(J, W());
  vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

  double F = numeric_limits<double>::max(), Fold;
  vector<double> Fxz(J);
  int i = 0;

  do
  {
    Fold = F;

    // Update Suff Stats and VBM for weights
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < J; ++j)
    {
      updateSS<C>(X[j], qZ[j], SSj[j], SS, sparse);
      vbmaximisationj<W>(SSj[j], wdists[j]);
    }

    // VBM for clusters
    #pragma omp parallel for schedule(guided)
    for (int k=0; k < K; ++k)
      cdists[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));

    // VBE
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < J; ++j)
      Fxz[j] = vbexpectation<W,C>(X[j], wdists[j], cdists, qZ[j], sparse);

    // Calculate free energy of model
    F = fenergy<W,C>(wdists, cdists, Fxz, SSj, SS);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > libcluster::FENGYDEL)
      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > libcluster::CONVERGE)
          && ( (i++ < maxit) || (maxit < 0) ) );

  return F;
}


/*  Find and remove all empty clusters. This is now necessary if we don't do an
 *    exhaustive search for the BEST cluster to split.
 *
 *    returns: true if any clusters have been deleted, false if all are kept.
 *    mutable: qZ may have columns deleted if there are empty clusters found.
 *    mutable: SSj if there are empty clusters found.
 *    mutable: SS if there are empty clusters found.
 */
bool clean (
    vector<MatrixXd>& qZ,              // Probabilities qZ
    vector<libcluster::SuffStat>& SSj, // Sufficient stats of groups
    libcluster::SuffStat& SS           // Sufficient stats
    )
{
  const int K = SS.getK(),
            J = qZ.size();
  ArrayXb kempty(K);

  // Look for empty sufficient statistics
  for (int k = 0; k < K; ++k)
    kempty(k) = SS.getN_k(k) < 1;

  // If everything is not empty, return false
  if ((kempty == false).all())
    return false;

  // Find location of empty and full clusters
  ArrayXi eidx, fidx;
  arrfind(kempty, eidx, fidx);

  // Delete empty cluster suff. stats.
  for (int i = eidx.size() - 1; i >= 0; --i)
  {
    SS.delk(eidx(i));
    for (int j = 0; j < J; ++j)
      SSj[j].delk(eidx(i));
  }

  // Delete empty cluster indicators by copying only full indicators
  const int newK = fidx.size();
  vector<MatrixXd> newqZ(J);

  for (int j = 0; j < J; ++j)
  {
    newqZ[j].setZero(qZ[j].rows(), newK);
    for (int k = 0; k < newK; ++k)
      newqZ[j].col(k) = qZ[j].col(fidx(k));
  }

  qZ = newqZ;

  return true;
}


/*  Search in an exhaustive fashion for a mixture split that lowers model free
 *    energy the most. If no splits are found which lower Free Energy, then
 *    false is returned, and qZ is not modified.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
#ifndef GREEDY_SPLIT
template <class W, class C> bool split_ex (
    const vector<MatrixXd>& X,               // Observations
    const vector<libcluster::SuffStat>& SSj, // Sufficient stats of groups
    const libcluster::SuffStat& SS,          // Sufficient stats
    const double F,                          // Current model free energy
    vector<MatrixXd>& qZ,                    // Probabilities qZ
    const bool sparse,                       // Do sparse updates to groups
    const bool verbose                       // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = SS.getK();

  // Pre allocate big objects for loops (this makes a runtime difference)
  double Fbest = numeric_limits<double>::infinity();
  vector<ArrayXi> mapidx(J, ArrayXi());
  vector<MatrixXd> qZref(J,MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd()),
                   qZbest;
  C csplit(SS.getprior(), X[0].cols());

  // Loop through each potential cluster in order and split it
  for (unsigned int k = 0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if (SS.getN_k(k) < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;
    csplit.update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));

    #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      mapidx[j] = partX(X[j], (qZ[j].col(k).array()>0.5), Xk[j]);  // Copy :-(
      Mtot += Xk[j].rows();

      // Initial cluster split
      ArrayXb splitk = csplit.splitobs(Xk[j]);
      qZref[j].setZero(Xk[j].rows(), 2);
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
    #pragma omp parallel for schedule(guided)
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
      cout << '=' << flush;

    // Record best splits so far
    if (Fsplit < Fbest)
    {
      qZbest = qZaug;
      Fbest  = Fsplit;
    }
  }

  // See if this split actually improves the model
  if ( (Fbest < F) && (abs((F-Fbest)/F) > CONVERGE) )
  {
    qZ = qZbest;
    return true;
  }
  else
    return false;
}
#endif


/*  Search in a greedy fashion for a mixture split that lowers model free
 *    energy, or return false. An attempt is made at looking for good, untried,
 *    split candidates first, as soon as a split canditate is found that lowers
 *    model F, it is returned. This may not be the "best" split, but it is
 *    certainly faster than an exhaustive search for the "best" split.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    mutable tally is a tally time a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
#ifdef GREEDY_SPLIT
template <class W, class C> bool split_gr (
    const vector<MatrixXd>& X,               // Observations
    const vector<libcluster::SuffStat>& SSj, // Sufficient stats of groups
    const libcluster::SuffStat& SS,          // Sufficient stats
    const double F,                          // Current model free energy
    vector<MatrixXd>& qZ,                    // Probabilities qZ
    vector<int>& tally,                      // Count of unsuccessful splits
    const bool sparse,                       // Do sparse updates to groups
    const bool verbose                       // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = SS.getK();

  // Split order chooser and cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);
  vector<C> csplit(K, C(SS.getprior(), X[0].cols()));

  // Get cluster parameters and their free energy
  #pragma omp parallel for schedule(guided)
  for (unsigned int k = 0; k < K; ++k)
  {
    csplit[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = csplit[k].fenergy();
  }

  // Get cluster likelihoods
  #pragma omp parallel for schedule(guided)
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
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector<ArrayXi> mapidx(J, ArrayXi());
  vector<MatrixXd> qZref(J,MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd());

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator i = ord.begin(); i < ord.end(); ++i)
  {
    const int k = i->k;

    ++tally[k]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (SS.getN_k(k) < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      mapidx[j] = partX(X[j], (qZ[j].col(k).array()>0.5), Xk[j]);  // Copy :-(
      Mtot += Xk[j].rows();

      // Initial cluster split
      ArrayXb splitk = csplit[k].splitobs(Xk[j]);
      qZref[j].setZero(Xk[j].rows(), 2);
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
    vbem<W,C>(Xk, qZref, SSgref, SSref, libcluster::SPLITITER, sparse);

    if (anyempty(SSref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    #pragma omp parallel for schedule(guided)
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
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > libcluster::CONVERGE) )
    {
      qZ = qZaug;
      tally[k] = 0;   // Reset tally if successfully split
      return true;
    }
  }

  // Failed to find splits
  return false;
}
#endif


/*  Bootstrap the clustering models. I.e. make sure we have reasonable starting
 *    assignments, qZ, depending on whether there are previous suff. stats.
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
  const unsigned int K = (SS.getK() < 1) ? 1 : SS.getK(),
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

      vbexpectation<W,C>(X[j], wdists[j], cdists, qZ[j], false);
    }
  }
  else  // This is an entirely new model to learn, start with one cluster
    for (unsigned int j = 0; j < J; ++j)
      qZ[j].setOnes(X[j].rows(), 1);
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
template <class W, class C> double modelselect (
    const vector<MatrixXd>& X,   // Observations
    vector<MatrixXd>& qZ,        // Observations to model mixture assignments
    vector<libcluster::SuffStat>& SSgroups, // Sufficient stats of groups
    libcluster::SuffStat& SS,    // Sufficient stats
    const bool sparse,           // Do sparse updates to groups
    const bool verbose,          // Verbose output
    const unsigned int nthreads  // Number of threads for OpenMP to use
    )
{
  if (nthreads < 1)
    throw invalid_argument("Must specify at least one thread for execution!");
  omp_set_num_threads(nthreads);

  // Bootstrap qZ, and sufficient statistics
  bootstrap<W,C>(X, SS, SSgroups, qZ);

  // Initialise free energy and other loop variables
  bool   issplit = true;
  double F;

  #ifdef GREEDY_SPLIT
  vector<int> tally;
  #endif

  // Main loop
  while (issplit == true)
  {
    // VBEM for all groups (throws runtime_error & invalid_argument)
    F = vbem<W,C>(X, qZ, SSgroups, SS, -1, sparse, verbose);

    // Remove any emtpy clusters
    bool remk = clean(qZ, SSgroups, SS);

    if ( (verbose == true) && (remk == true) )
      cout << 'x' << flush;

    // Start cluster splitting
    if (verbose == true)
      cout << '<' << flush;  // Notify start splitting

    // Search for best split, augment qZ if found one
    #ifdef GREEDY_SPLIT
    issplit = split_gr<W,C>(X, SSgroups, SS, F, qZ, tally, sparse, verbose);
    #else
    issplit = split_ex<W,C>(X, SSgroups, SS, F, qZ, sparse, verbose);
    #endif

    if (verbose == true)
      cout << '>' << endl;   // Notify end splitting
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of clusters = " << SS.getK() << endl;
    cout << "Free energy = " << F << endl;
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
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning VDP..." << endl; // Print start

  // Make temporary vectors of data to use with modelselect()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F = modelselect<StickBreak, GaussWish>(vecX, vecqZ, SSgroup, SS, false,
                                                verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                        // copy :-(
  return F;
}


double libcluster::learnBGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    SuffStat& SS,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning Bayesian GMM..." << endl; // Print start

  // Make temporary vectors of data to use with modelselect()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<libcluster::SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F = modelselect<Dirichlet, GaussWish>(vecX, vecqZ, SSgroup, SS, false,
                                               verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copy :-(
  return F;
}


double libcluster::learnDGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    SuffStat& SS,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning Bayesian diagonal GMM..." << endl; // Print start

  // Make temporary vectors of data to use with modelselect()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<libcluster::SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F = modelselect<Dirichlet, NormGamma>(vecX, vecqZ, SSgroup, SS, false,
                                               verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copy :-(
  return F;
}


double libcluster::learnBEMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    SuffStat& SS,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if ((X.array() < 0).any() == true)
    throw invalid_argument("X has to be in the range [0, inf)!");

  if (verbose == true)
    cout << "Learning Bayesian EMM..." << endl; // Print start

  // Make temporary vectors of data to use with modelselect()
  vector<MatrixXd> vecX(1, X);          // copy :-(
  vector<MatrixXd> vecqZ;
  vector<libcluster::SuffStat> SSgroup(1, SuffStat(SS.getprior()));

  // Perform model learning and selection
  double F = modelselect<Dirichlet, ExpGamma>(vecX, vecqZ, SSgroup, SS, false,
                                              verbose, nthreads);

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
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "GMC..." << endl;

  return modelselect<GDirichlet, GaussWish>(X, qZ, SSgroups, SS, sparse,
                                            verbose, nthreads);
}


double libcluster::learnSGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "Symmetric GMC..." << endl;

  return modelselect<Dirichlet, GaussWish>(X, qZ, SSgroups, SS, sparse, verbose,
                                           nthreads);
}


double libcluster::learnDGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "Diagonal GMC..." << endl;

  return modelselect<GDirichlet, NormGamma>(X, qZ, SSgroups, SS, sparse,
                                            verbose, nthreads);
}


double libcluster::learnEGMC (
    const vector<MatrixXd>& X,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSgroups,
    libcluster::SuffStat& SS,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Check for negative inputs
  for (unsigned int j = 0; j < X.size(); ++j)
    if ((X[j].array() < 0).any() == true)
      throw invalid_argument("X has to be in the range [0, inf)!");

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "Exponential GMC..." << endl;

  return modelselect<GDirichlet, ExpGamma>(X, qZ, SSgroups, SS, sparse, verbose,
                                           nthreads);
}
