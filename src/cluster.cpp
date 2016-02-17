/*
 * libcluster -- A collection of hierarchical Bayesian clustering algorithms.
 * Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
 *
 * This file is part of libcluster.
 *
 * libcluster is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * libcluster is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcluster. If not, see <http://www.gnu.org/licenses/>.
 */

// TODO:
//  - sparse updates sometimes create positive free energy steps.

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
using namespace libcluster;


//
// Variational Bayes Private Functions
//


/* Update the group and model sufficient statistics based on assignments qZj.
 *
 *  mutable: the clusters (add sufficient stats).
 *  returns: the number of observations in each cluster for this groups.
 */
template <class C> ArrayXd updateSS (
    const MatrixXd& Xj,   // Observations in group j
    const MatrixXd& qZj,  // Observations to group mixture assignments
    vector<C>& clusters,  // Cluster Distributions
    const bool sparse     // Do sparse updates to groups
    )
{
  const unsigned int K = qZj.cols();

  const ArrayXd Njk = qZj.colwise().sum();  // count obs. in this group
  ArrayXi Kful = ArrayXi::Zero(1),          // Initialise and set K = 1 defaults
          Kemp = ArrayXi::Zero(0);

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((Njk >= ZEROCUTOFF), Kful, Kemp);

  const unsigned int nKful = Kful.size();

  // Sufficient statistics - with observations
  for (unsigned int k = 0; k < nKful; ++k)
  {
    #pragma omp critical
    clusters[Kful(k)].addobs(qZj.col(Kful(k)), Xj);
  }

  return Njk;
}


/* The Variational Bayes Expectation step for each group.
 *
 *  mutable: Group assignment probabilities, qZj
 *  returns: The complete-data (X,Z) free energy E[log p(X,Z)/q(Z)] for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class W, class C> double vbexpectation (
    const MatrixXd& Xj,         // Observations in group J
    const W& weights,           // Group Weight parameter distribution
    const vector<C>& clusters,  // Cluster parameter distributions
    MatrixXd& qZj,              // Observations to group mixture assignments
    const bool sparse           // Do sparse updates to groups
    )
{
  const int K  = clusters.size(),
            Nj = Xj.rows();

  // Get log marginal weight likelihoods
  const ArrayXd E_logZ = weights.Elogweight();

  // Initialise and set K = 1 defaults for cluster counts
  ArrayXi Kful = ArrayXi::Zero(1), Kemp = ArrayXi::Zero(0);

  // Find empty clusters if sparse
  if ( (sparse == false) && (K > 1) )
    Kful = ArrayXi::LinSpaced(Sequential, K, 0, K-1);
  else if (sparse == true)
    arrfind((weights.getNk() >= ZEROCUTOFF), Kful, Kemp);

  const int nKful = Kful.size(),
            nKemp = Kemp.size();

  // Find Expectations of log joint observation probs -- allow sparse evaluation
  MatrixXd logqZj(Nj, nKful);

  for (int k = 0; k < nKful; ++k)
    logqZj.col(k) = E_logZ(Kful(k)) + clusters[Kful(k)].Eloglike(Xj).array();

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
 */
template <class W, class C> double fenergy (
    const vector<W>& weights,   // Weight parameter distributions
    const vector<C>& clusters,  // Cluster parameter distributions
    const double Fxz            // Free energy from data log-likelihood
    )
{
  const int K = clusters.size(),
            J = weights.size();

  // Free energy of the weight parameter distributions
  double Fw = 0;
  for (int j = 0; j < J; ++j)
    Fw += weights[j].fenergy();

  // Free energy of the cluster parameter distributionsreturn
  double Fc = 0;
  for (int k = 0; k < K; ++k)
    Fc += clusters[k].fenergy();

  return Fc + Fw + Fxz;
}


/* Variational Bayes EM for all group mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: the group weight distributions
 *  mutable: the cluster distributions
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class C> double vbem (
    const vMatrixXd& X,         // Observations
    vMatrixXd& qZ,              // Observations to model mixture assignments
    vector<W>& weights,         // Group weight distributions
    vector<C>& clusters,        // Cluster Distributions
    const double clusterprior,  // Prior value for cluster distributions
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false  // Verbose output (default false)
    )
{
  const int J = X.size(),
            K = qZ[0].cols();

  // Construct (empty) parameters
  weights.resize(J, W());
  clusters.resize(K, C(clusterprior, X[0].cols()));

  double F = numeric_limits<double>::max(), Fold;
  int i = 0;

  do
  {
    Fold = F;

    // Clear Suffient Statistics
    for (int k = 0; k < K; ++k)
      clusters[k].clearobs();

    // Update Suff Stats and VBM for weights
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < J; ++j)
    {
      ArrayXd Njk = updateSS<C>(X[j], qZ[j], clusters, sparse);
      weights[j].update(Njk);
    }

    // VBM for clusters
    #pragma omp parallel for schedule(guided)
    for (int k = 0; k < K; ++k)
      clusters[k].update();

    // VBE
    double Fz = 0;
    #pragma omp parallel for schedule(guided) reduction(+ : Fz)
    for (int j = 0; j < J; ++j)
      Fz += vbexpectation<W,C>(X[j], weights[j], clusters, qZ[j], sparse);

    // Calculate free energy of model
    F = fenergy<W,C>(weights, clusters, Fz);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > FENGYDEL)
      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > CONVERGE)
          && ( (i++ < maxit) || (maxit < 0) ) );

  return F;
}


//
//  Model Selection and Heuristics Private Functions
//


/*  Search in an exhaustive fashion for a mixture split that lowers model free
 *    energy the most. If no splits are found which lower Free Energy, then
 *    false is returned, and qZ is not modified.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
#ifdef EXHAUST_SPLIT
template <class W, class C> bool split_ex (
    const vMatrixXd& X,         // Observations
    const vector<C>& clusters,  // Cluster Distributions
    vMatrixXd& qZ,              // Probabilities qZ
    const double F,             // Current model free energy
    const int maxclusters,      // maximum number of clusters to search for
    const bool sparse,          // Do sparse updates to groups
    const bool verbose          // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = clusters.size();

  // Check if we have reached the max number of clusters
  if ( ((signed) K >= maxclusters) && (maxclusters >= 0) )
      return false;

  // Pre allocate big objects for loops (this makes a runtime difference)
  double Fbest = numeric_limits<double>::infinity();
  vector<ArrayXi> mapidx(J, ArrayXi());
  vMatrixXd qZref(J,MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd()), qZbest;

  // Loop through each potential cluster in order and split it
  for (unsigned int k = 0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if (clusters[k].getN() < 4)
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
      ArrayXb splitk = clusters[k].splitobs(Xk[j]);
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
    vector<W> wspl;
    vector<C> cspl;
    vbem<W,C>(Xk, qZref, wspl, cspl, clusters[0].getprior(), SPLITITER, sparse);

    if (anyempty<C>(cspl) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    #pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < J; ++j)
      qZaug[j] = augmentqZ(k, mapidx[j], (qZref[j].col(1).array()>0.5), qZ[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    double Fsplit = vbem<W,C>(X, qZaug, wspl, cspl,  clusters[0].getprior(), 1,
                              sparse);

    if (anyempty<C>(cspl) == true) // One cluster only
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
#ifndef EXHAUST_SPLIT
template <class W, class C> bool split_gr (
    const vMatrixXd& X,         // Observations
    const vector<W>& weights,   // Group weight distributions
    const vector<C>& clusters,  // Cluster Distributions
    vMatrixXd& qZ,              // Probabilities qZ
    vector<int>& tally,         // Count of unsuccessful splits
    const double F,             // Current model free energy
    const int maxclusters,      // maximum number of clusters to search for
    const bool sparse,          // Do sparse updates to groups
    const bool verbose          // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = clusters.size();

  // Check if we have reached the max number of clusters
  if ( ((signed) K >= maxclusters) && (maxclusters >= 0) )
      return false;

  // Split order chooser and cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);

  // Get cluster parameters and their free energy
  #pragma omp parallel for schedule(guided)
  for (unsigned int k = 0; k < K; ++k)
  {
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = clusters[k].fenergy();
  }

  // Get cluster likelihoods
  #pragma omp parallel for schedule(guided)
  for (unsigned int j = 0; j < J; ++j)
  {
    // Get cluster weights
    ArrayXd logpi = weights[j].Elogweight();

    // Add in cluster log-likelihood, weighted by responsability
    for (unsigned int k = 0; k < K; ++k)
    {
      double LL = qZ[j].col(k).dot((logpi(k)
                                + clusters[k].Eloglike(X[j]).array()).matrix());

      #pragma omp atomic
      ord[k].Fk -= LL;
    }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector<ArrayXi> mapidx(J, ArrayXi());
  vMatrixXd qZref(J, MatrixXd()), qZaug(J,MatrixXd()), Xk(J,MatrixXd());

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator i = ord.begin(); i < ord.end(); ++i)
  {
    const int k = i->k;

    ++tally[k]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (clusters[k].getN() < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      mapidx[j] = partobs(X[j], (qZ[j].col(k).array()>0.5), Xk[j]);  // Copy :-(
      Mtot += Xk[j].rows();

      // Initial cluster split
      ArrayXb splitk = clusters[k].splitobs(Xk[j]);
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
    vector<W> wspl;
    vector<C> cspl;
    vbem<W,C>(Xk, qZref, wspl, cspl, clusters[0].getprior(), SPLITITER, sparse);

    if (anyempty<C>(cspl) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    #pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < J; ++j)
      qZaug[j] = auglabels(k, mapidx[j], (qZref[j].col(1).array()>0.5), qZ[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    double Fsplit = vbem<W,C>(X, qZaug, wspl, cspl,  clusters[0].getprior(), 1,
                              sparse);

    if (anyempty<C>(cspl) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > CONVERGE) )
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


/*  Find and remove all empty clusters.
 *
 *    returns: true if any clusters have been deleted, false if all are kept.
 *    mutable: qZ may have columns deleted if there are empty clusters found.
 *    mutable: weights if there are empty clusters found.
 *    mutable: clusters if there are empty clusters found.
 */
template <class W, class C> bool prune_clusters (
    vMatrixXd& qZ,        // Probabilities qZ
    vector<W>& weights,   // weights distributions
    vector<C>& clusters,  // cluster distributions
    bool verbose = false  // print status
    )
{
  const unsigned int K = clusters.size(),
                     J = qZ.size();

  // Look for empty clusters
  ArrayXd Nk(K);
  for (unsigned int k= 0; k < K; ++k)
    Nk(k) = clusters[k].getN();

  // Find location of empty and full clusters
  ArrayXi eidx, fidx;
  arrfind(Nk.array() < ZEROCUTOFF, eidx, fidx);
  const unsigned int nempty = eidx.size();

  // If everything is not empty, return false
  if (nempty == 0)
    return false;

  if (verbose == true)
    cout << '*' << flush;

  // Delete empty cluster suff. stats.
  for (int i = (nempty - 1); i >= 0; --i)
    clusters.erase(clusters.begin() + eidx(i));

  // Delete empty cluster indicators by copying only full indicators
  const unsigned int newK = fidx.size();
  vMatrixXd newqZ(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    newqZ[j].setZero(qZ[j].rows(), newK);
    for (unsigned int k = 0; k < newK; ++k)
      newqZ[j].col(k) = qZ[j].col(fidx(k));

    weights[j].update(newqZ[j].colwise().sum()); // new weights
  }

  qZ = newqZ;

  return true;
}


/* The model selection algorithm for a grouped mixture model.
 *
 *  returns: Free energy of the final model
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: the group weight distributions
 *  mutable: the cluster distributions
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if free energy increases.
 */
template <class W, class C> double cluster (
    const vMatrixXd& X,           // Observations
    vMatrixXd& qZ,                // Observations to model mixture assignments
    vector<W>& weights,           // Group weight distributions
    vector<C>& clusters,          // Cluster Distributions
    const double clusterprior,    // Prior value for cluster distributions
    const int maxclusters,        // Maximum number of clusters to search for
    const bool sparse,            // Do sparse updates to groups
    const bool verbose,           // Verbose output
    const unsigned int nthreads   // Number of threads for OpenMP to use
    )
{
  if (nthreads < 1)
    throw invalid_argument("Must specify at least one thread for execution!");
  omp_set_num_threads(nthreads);

  const unsigned int J = X.size();

  // Initialise indicator variables to just one cluster
  qZ.resize(J);
  for (unsigned int j = 0; j < J; ++j)
    qZ[j].setOnes(X[j].rows(), 1);

  // Initialise free energy and other loop variables
  bool issplit = true;
  double F;

  #ifndef EXHAUST_SPLIT
  vector<int> tally;
  #endif

  // Main loop
  while (issplit == true)
  {
    // VBEM for all groups (throws runtime_error & invalid_argument)
    F = vbem<W,C>(X, qZ, weights, clusters, clusterprior, -1, sparse, verbose);

    // Remove any empty clusters
    prune_clusters<W,C>(qZ, weights, clusters, verbose);

    // Start cluster splitting
    if (verbose == true)
      cout << '<' << flush;  // Notify start splitting

    // Search for best split, augment qZ if found one
    #ifdef EXHAUST_SPLIT
    issplit = split_ex<W,C>(X, clusters, qZ, F, maxclusters, sparse, verbose);
    #else
    issplit = split_gr<W,C>(X, weights, clusters, qZ, tally, F, maxclusters,
                            sparse, verbose);
    #endif

    if (verbose == true)
      cout << '>' << endl;   // Notify end splitting
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of clusters = " << clusters.size() << endl;
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
    StickBreak& weights,
    vector<GaussWish>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning VDP..." << endl; // Print start

  // Make temporary vectors of data to use with cluster()
  vMatrixXd vecX(1, X);                 // copies :-(
  vMatrixXd vecqZ;
  vector<StickBreak> vecweights(1, weights);

  // Perform model learning and selection
  double F = cluster<StickBreak, GaussWish>(vecX, vecqZ, vecweights, clusters,
                                        clusterprior, maxclusters, false,
                                        verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                        // copies :-(
  weights = vecweights[0];
  return F;
}


double libcluster::learnBGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    Dirichlet& weights,
    vector<GaussWish>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning Bayesian GMM..." << endl; // Print start

  // Make temporary vectors of data to use with cluster()
  vMatrixXd vecX(1, X);                   // copies :-(
  vMatrixXd vecqZ;
  vector<Dirichlet> vecweights(1, weights);

  // Perform model learning and selection
  double F = cluster<Dirichlet, GaussWish>(vecX, vecqZ, vecweights, clusters,
                                        clusterprior, maxclusters, false,
                                        verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copies :-(
  weights = vecweights[0];
  return F;
}


double libcluster::learnDGMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    Dirichlet& weights,
    vector<NormGamma>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if (verbose == true)
    cout << "Learning Bayesian diagonal GMM..." << endl; // Print start

  // Make temporary vectors of data to use with cluster()
  vMatrixXd vecX(1, X);                   // copies :-(
  vMatrixXd vecqZ;
  vector<Dirichlet> vecweights(1, weights);

  // Perform model learning and selection
  double F = cluster<Dirichlet, NormGamma>(vecX, vecqZ, vecweights, clusters,
                                        clusterprior, maxclusters, false,
                                        verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copies :-(
  weights = vecweights[0];
  return F;
}


double libcluster::learnBEMM (
    const MatrixXd& X,
    MatrixXd& qZ,
    Dirichlet& weights,
    vector<ExpGamma>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool verbose,
    const unsigned int nthreads
    )
{
  if ((X.array() < 0).any() == true)
    throw invalid_argument("X has to be in the range [0, inf)!");

  if (verbose == true)
    cout << "Learning Bayesian EMM..." << endl; // Print start

  // Make temporary vectors of data to use with cluster()
  vMatrixXd vecX(1, X);                   // copies :-(
  vMatrixXd vecqZ;
  vector<Dirichlet> vecweights(1, weights);

  // Perform model learning and selection
  double F = cluster<Dirichlet, ExpGamma>(vecX, vecqZ, vecweights, clusters,
                                        clusterprior, maxclusters, false,
                                        verbose, nthreads);

  // Return final Free energy and qZ
  qZ = vecqZ[0];                          // copies :-(
  weights = vecweights[0];
  return F;
}


double libcluster::learnGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    vector<GDirichlet>& weights,
    vector<GaussWish>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "GMC..." << endl;

  return cluster<GDirichlet, GaussWish>(X, qZ, weights, clusters, clusterprior,
                                        maxclusters, sparse, verbose,
                                        nthreads);
}


double libcluster::learnSGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    vector<Dirichlet>& weights,
    vector<GaussWish>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "Symmetric GMC..." << endl;

  return cluster<Dirichlet, GaussWish>(X, qZ, weights, clusters, clusterprior,
                                       maxclusters, sparse, verbose, nthreads);
}


double libcluster::learnDGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    vector<GDirichlet>& weights,
    vector<NormGamma>& clusters,
    const double clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const unsigned int nthreads
    )
{
  string spnote = (sparse == true) ? "(sparse) " : "";

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << spnote << "Diagonal GMC..." << endl;

  return cluster<GDirichlet, NormGamma>(X, qZ, weights, clusters, clusterprior,
                                        maxclusters, sparse, verbose,
                                        nthreads);
}


double libcluster::learnEGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    vector<GDirichlet>& weights,
    vector<ExpGamma>& clusters,
    const double clusterprior,
    const int maxclusters,
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

  return cluster<GDirichlet, ExpGamma>(X, qZ, weights, clusters, clusterprior,
                                       maxclusters, sparse, verbose, nthreads);
}
