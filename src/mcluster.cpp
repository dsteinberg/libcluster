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

#include <limits>
#include "libcluster.h"
#include "probutils.h"
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
//  Variational Bayes Private Functions
//

/* The Variational Bayes Expectation step for weights in each group.
 *
 *  mutable: Top-level cluster assignment probabilities, qYj
 *  returns: The complete-data free energy, Y and Y+Z dep. terms, for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class WJ, class WT, class CT> double vbeY (
    const MatrixXd& Wj,           // Top-level observations for group j
    const vMatrixXd& qZj,         // Bottom-level cluster labels for group j
    const WJ& weightsj,           // Group top-level cluster weights
    const vector<WT>& weights_t,  // Bottom-level cluster proportion/paramters
    const vector<CT>& clusters_t, // Top-level cluster parameters
    MatrixXd& qYj                 // Top-level cluster assignments for group j
    )
{
  const unsigned int T  = weights_t.size(),
                     Ij = qZj.size(),
                     K  = qZj[0].cols();

  // No observations (may happen when splitting)
  if (Ij == 0)
    return 0;

  // Get log marginal weight likelihoods
  const ArrayXd E_logwj = weightsj.Elogweight();

  MatrixXd Njik(Ij, K), logqYj(Ij, T);
  ArrayXXd qZjiLike(Ij, T);

  // Get bottom-level cluster counts per "document/image"
  for (unsigned int i = 0; i < Ij; ++i)
    Njik.row(i) = qZj[i].colwise().sum();

  // Find Expectations of log joint observation probs
  for (unsigned int t = 0; t < T; ++t)
  {
    qZjiLike.col(t) = Njik * weights_t[t].Elogweight().matrix();
    logqYj.col(t)   = qZjiLike.col(t) + E_logwj(t)
                      + clusters_t[t].Eloglike(Wj).array();
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZyj = logsumexp(logqYj);

  // Normalise and Compute Responsibilities
  qYj = (logqYj.colwise() - logZyj).array().exp().matrix();

  return ((qYj.array() * qZjiLike).rowwise().sum() - logZyj.array()).sum();
}


/* The Variational Bayes Expectation step for clusters in each "document", ji.
 *
 *  mutable: Bottom-level cluster assignment probabilities, qZji
 *  returns: The complete-data free energy, Z dep. terms, for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class WT, class CK> double vbeZ (
    const MatrixXd& Xji,          // Observations in i in group j
    const RowVectorXd& qYji,      // Top-level cluster assignment
    const vector<WT>& weights_t,  // Top-level cluster parameters
    const vector<CK>& clusters_k, // Bottom-level cluster parameters
    MatrixXd& qZji                // Observation to cluster assignments
    )
{
  const int K   = clusters_k.size(),
            Nji = Xji.rows(),
            T   = weights_t.size();

  // No observations (may happen when splitting)
  if (Nji == 0)
    return 0;

  // Make top-level cluster global weights from weighted label parameters
  RowVectorXd E_logqYljt = RowVectorXd::Zero(K);

  for (int t = 0; t < T; ++t)
    E_logqYljt.noalias() += qYji(t) * weights_t[t].Elogweight().matrix();

  // Find Expectations of log joint observation probs
  MatrixXd logqZji = MatrixXd::Zero(Nji, K);

  for (int k = 0; k < K; ++k)
    logqZji.col(k) = E_logqYljt(k) + clusters_k[k].Eloglike(Xji).array();

  // Log normalisation constant of log observation likelihoods
  const VectorXd logZzji = logsumexp(logqZji);

  // Normalise and Compute Responsibilities
  qZji = (logqZji.colwise() - logZzji).array().exp().matrix();

  return -logZzji.sum();
}


/* Calculates the free energy lower bound for the model parameter distributions.
 *
 *  returns: the free energy of the model
 */
template <class WJ, class WT, class CT, class CK> double fenergy (
    const vector<WJ>& weights_j,  // Group top-level cluster weights
    const vector<WT>& weights_t,  // Top-level cluster proportion parameters
    const vector<CT>& clusters_t, // Top-level cluster other parameters
    const vector<CK>& clusters_k, // Bottom-level cluster parameters
    const double Fyz,             // Free energy Y and cross Y-Z terms
    const double Fz               // Free energy Z terms
    )
{
  const int T = weights_t.size(),
            K = clusters_k.size(),
            J = weights_j.size();

  // Class parameter free energy
  double Ft = 0;
  for (int t = 0; t < T; ++t)
    Ft += weights_t[t].fenergy() + clusters_t[t].fenergy();

  // Cluster parameter free energy
  double Fk = 0;
  for (int k = 0; k < K; ++k)
    Fk += clusters_k[k].fenergy();

  // Weight parameter free energy
  double Fw = 0;
  for (int j = 0; j < J; ++j)
    Fw += weights_j[j].fenergy();

  return Fw + Ft + Fk + Fyz + Fz;
}


/* Variational Bayes EM. 
 *
 *  returns: Free energy of the whole model.
 *  mutable: the bottom-level cluster indicators, qZ
 *  mutable: the top-level cluster indicators, qY
 *  mutable: model parameters weights_j, weights_t, clusters_k, clusters_t
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
template<class WJ, class WT, class CT, class CK> double vbem (
    const vMatrixXd& W,           // Top-level observations
    const vvMatrixXd& X,          // Bottom-level observations
    vMatrixXd& qY,                // Top-level labels
    vvMatrixXd& qZ,               // Bottom-level cluster labels
    vector<WJ>& weights_j,        // Group top-level cluster weights
    vector<WT>& weights_t,        // Top-level proportion cluster parameters
    vector<CT>& clusters_t,       // Top-level other cluster parameters
    vector<CK>& clusters_k,       // Bottom-level cluster parameters
    const double prior_t,         // Top-level cluster prior
    const double prior_k,         // Bottom-level cluster prior
    const int maxit = -1,         // Max VBEM iterations (-1 = no max, default)
    const bool verbose = false    // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = qZ[0][0].cols(),
                     T = qY[0].cols();

  // Construct (empty) parameters
  weights_j.resize(J, WJ());
  weights_t.resize(T, WT());
  clusters_t.resize(T, CT(prior_t, W[0].cols()));
  clusters_k.resize(K, CK(prior_k, X[0][0].cols()));

  // Other loop variables for initialisation
  int it = 0;
  double F = numeric_limits<double>::max(), Fold;

  do
  {
    Fold = F;

    MatrixXd Ntk = MatrixXd::Zero(T, K); // Clear Sufficient Stats

    // VBM for top-level cluster weights
    #pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Accumulate suff. stats for bottom-level cluster counts
      for (unsigned int i = 0; i < X[j].size(); ++i)
      {
        MatrixXd Ntkji = qY[j].row(i).transpose() * qZ[j][i].colwise().sum();
        #pragma omp critical
        Ntk += Ntkji;
      }

      weights_j[j].update(qY[j].colwise().sum());
    }

    // VBM for top-level cluster parameters and proportions
    #pragma omp parallel for schedule(guided)
    for (unsigned int t = 0; t < T; ++t)
    {
      clusters_t[t].clearobs();                  // Clear Sufficient Stats

      for (unsigned int j = 0; j < J; ++j)       // Accumulate sufficient stats
        clusters_t[t].addobs(qY[j].col(t), W[j]);

      weights_t[t].update(Ntk.row(t));           // Bottom-level cluster counts.
      clusters_t[t].update();
    }

    // VBM for bottom-level cluster parameters
    #pragma omp parallel for schedule(guided)
    for (unsigned int k = 0; k < K; ++k)
    {
      clusters_k[k].clearobs();                  // Clear Sufficient Stats

      for (unsigned int j = 0; j < J; ++j)       // Accumulate sufficient stats
        for (unsigned int i = 0; i < X[j].size(); ++i)
          clusters_k[k].addobs(qZ[j][i].col(k), X[j][i]);

      clusters_k[k].update();                    // Bottom-level observations
    }

    // Free energy data fit term accumulators
    double Fz = 0, Fyz = 0;

    // VBE for top-level cluster indicators
    #pragma omp parallel for schedule(guided) reduction(+ : Fyz)
    for (unsigned int j = 0; j < J; ++j)
      Fyz += vbeY<WJ,WT,CT>(W[j], qZ[j], weights_j[j], weights_t, clusters_t,
                            qY[j]);

    // VBE for bottom-level cluster indicators
    for (unsigned int j = 0; j < J; ++j)
    {
      #pragma omp parallel for schedule(guided) reduction(+ : Fz)
      for (unsigned int i = 0; i < X[j].size(); ++i)
        Fz += vbeZ<WT,CK>(X[j][i], qY[j].row(i), weights_t, clusters_k, 
                          qZ[j][i]);
    }

    // Calculate free energy of model
    F = fenergy<WJ,WT,CT,CK>(weights_j, weights_t, clusters_t, clusters_k, Fyz, 
                            Fz);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > libcluster::FENGYDEL)
      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > libcluster::CONVERGE)
          && ( (++it < maxit) || (maxit < 0) ) );

  return F;
}


//
//  Model Selection and Heuristics Private Functions
//

/*  Search in a greedy fashion for a mixture split that lowers model free
 *    energy, or return false. An attempt is made at looking for good, untried,
 *    split candidates first, as soon as a split canditate is found that lowers
 *    model F, it is returned. This may not be the "best" split, but it is
 *    certainly faster than an exhaustive search for the "best" split.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    mutable: qY is updated if a new split if one is found, otherwise left
 *    mutable tally is a tally of times a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
template <class WJ, class WT, class CT, class CK> bool ssplit (
    const vMatrixXd& W,             // Top-level observations
    const vvMatrixXd& X,            // Bottom-level observations
    const vector<CT>& clusters_t,   // Top-level cluster Distributions
    const vector<CK>& clusters_k,   // Bottom-level cluster Distributions
    vMatrixXd& qY,                  // Top-level cluster labels qY
    vvMatrixXd& qZ,                 // Bottom-level Cluster labels qZ
    vector<int>& tally,             // Count of unsuccessful splits
    const double F,                 // Current model free energy
    const int maxK,                 // max number of (bottom) clusters
    const bool verbose              // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = clusters_k.size();

  // Check if we have reached the max number of clusters
  if ( ((signed) K >= maxK) && (maxK >= 0) )
      return false;

  // Split order chooser and bottom-level cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);

  // Get cluster parameters and their free energy
  for (unsigned int k = 0; k < K; ++k)
  {
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = clusters_k[k].fenergy();
  }

  // Get bottom-level cluster likelihoods
  for (unsigned int j = 0; j < J; ++j)
  {
    // Add in cluster log-likelihood, weighted by global responsability
    #pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < X[j].size(); ++i)
      for (unsigned int k = 0; k < K; ++k)
      {
        double LL = qZ[j][i].col(k).dot(clusters_k[k].Eloglike(X[j][i]));

        #pragma omp atomic
        ord[k].Fk -= LL;
      }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector< vector<ArrayXi> > mapidx(J);
  vvMatrixXd qZref(J), qZaug(J), Xk(J);

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator ko = ord.begin(); ko < ord.end(); ++ko)
  {
    const int k = ko->k;

    ++tally[k]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (clusters_k[k].getN() < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    for (unsigned int j = 0; j < J; ++j)
    {
      mapidx[j].resize(X[j].size());
      qZref[j].resize(X[j].size());
      qZaug[j].resize(X[j].size());
      Xk[j].resize(X[j].size());

      #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
      for (unsigned int i = 0; i < X[j].size(); ++i)
      {
        // Make COPY of the observations with only relevant data points, p > 0.5
        mapidx[j][i] = partobs(X[j][i], (qZ[j][i].col(k).array()>0.5),
                               Xk[j][i]);
        Mtot += Xk[j][i].rows();

        // Initial cluster split
        ArrayXb splitk = clusters_k[k].splitobs(Xk[j][i]);
        qZref[j][i].setZero(Xk[j][i].rows(), 2);
        qZref[j][i].col(0) = (splitk == true).cast<double>();
        qZref[j][i].col(1) = (splitk == false).cast<double>();

        // keep a track of number of splits
        scount += splitk.count();
      }
    }

    // Don't waste time with clusters that haven't been split sufficiently
    if ( (scount < 2) || (scount > (Mtot-2)) )
      continue;

    // Refine the split
    vector<WJ> iwspl;
    vector<CT> icspl;
    vector<WT> swspl;
    vector<CK> scspl;
    vMatrixXd qYaug = qY;                             // Copy :-(
    vbem<WJ,WT,CT,CK>(W, Xk, qYaug, qZref, iwspl, swspl, icspl, scspl,
                clusters_t[0].getprior(), clusters_k[0].getprior(), SPLITITER);

    if (anyempty<CK>(scspl) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    for (unsigned int j = 0; j < J; ++j)
    {
      #pragma omp parallel for schedule(guided)
      for (unsigned int i = 0; i < X[j].size(); ++i)
        qZaug[j][i] = auglabels(k, mapidx[j][i],
                                (qZref[j][i].col(1).array() > 0.5), qZ[j][i]);
    }

    // Calculate free energy of this split with ALL data (and refine a bit)
    qYaug = qY;                             // Copy :-(
    double Fs = vbem<WJ,WT,CT,CK>(W, X, qYaug, qZaug, iwspl, swspl, icspl,
                  scspl, clusters_t[0].getprior(), clusters_k[0].getprior(), 1);

    if (anyempty<CK>(scspl) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fs < F) && (abs((F-Fs)/F) > CONVERGE) )
    {
      qY = qYaug;
      qZ = qZaug;
      tally[k] = 0;   // Reset tally if successfully split
      return true;
    }
  }

  // Failed to find splits
  return false;
}


/*  Find and remove all empty top-level clusters.
 *
 *    returns: true if any clusters have been deleted, false if all are kept.
 *    mutable: qY may have columns deleted if there are empty clusters found.
 *    mutable: weights_t if there are empty top-level clusters found.
 *    mutable: clusters_t if there are empty top-level clusters found.
 */
template <class WT, class CT> bool prune_clusters_t (
    vMatrixXd& qY,          // Probabilities qY
    vector<WT>& weights_t,  // Top-level bottom-level cluster proportions
    vector<CT>& clusters_t, // Top-level clusters
    bool verbose = false    // print status
    )
{
  const unsigned int T = weights_t.size(),
                     J = qY.size();

  // Look for empty clusters
  ArrayXd Nt(T);
  for (unsigned int t = 0; t < T; ++t)
    Nt(t) = weights_t[t].getNk().sum();

  // Find location of empty and full clusters
  ArrayXi eidx, fidx;
  arrfind(Nt.array() < 1, eidx, fidx);
  const unsigned int nempty = eidx.size();

  // If everything is not empty, return false
  if (nempty == 0)
    return false;

  if (verbose == true)
    cout << '*' << flush;

  // Delete empty clusters
  for (int i = (nempty - 1); i >= 0; --i)
  {
    weights_t.erase(weights_t.begin() + eidx(i));
    clusters_t.erase(clusters_t.begin() + eidx(i));
  }

  // Delete empty cluster indicators by copying only full indicators
  const unsigned int newT = fidx.size();
  vMatrixXd newqY(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    newqY[j].setZero(qY[j].rows(), newT);
    for (unsigned int t = 0; t < newT; ++t)
      newqY[j].col(t) = qY[j].col(fidx(t));
  }

  qY = newqY;

  return true;
}


/* The model selection algorithm
 *
 *  returns: Free energy of the final model
 *  mutable: qY the probabilistic top-level cluster assignments
 *  mutable: qZ the probabilistic bottom-level cluster assignments
 *  mutable: The top-level clusters and weights
 *  mutable: The bottom-level clusters and bottom-level cluster weights
 *  throws: invalid_argument from other functions
 *  throws: runtime_error if free energy increases
 */
template<class WJ, class WT, class CT, class CK> double mcluster (
    const vMatrixXd& W,           // Top-level observations
    const vvMatrixXd& X,          // Bottom-level observations
    vMatrixXd& qY,                // Top-level labels
    vvMatrixXd& qZ,               // Bottom-level labels
    vector<WJ>& weights_j,        // Group top-level cluster weights
    vector<WT>& weights_t,        // Tope-level proportion cluster parameters 
    vector<CT>& clusters_t,       // Top-level cluster parameters
    vector<CK>& clusters_k,       // Bottom-level cluster parameters
    const double prior_t,         // Top-level cluster prior
    const double prior_k,         // Bottom-level cluster prior
    const unsigned int maxT,      // Truncation level for top-level clusters
    const int maxK,               // max number of (bottom) clusters
    const bool verbose,           // Verbose output
    const unsigned int nthreads   // Number of threads for OpenMP to use
    )
{
  if (nthreads < 1)
    throw invalid_argument("Must specify at least one thread for execution!");
  omp_set_num_threads(nthreads);

  // Do some observation validity checks
  if (W.size() != X.size()) // Same number of groups in observations
    throw invalid_argument("W and X need to have the same number of groups!");

  const unsigned int J = W.size();

  for (unsigned int j = 0; j < J; ++j) // Same number of images/docs in groups
    if ((unsigned) W[j].rows() != X[j].size())
      throw invalid_argument("W and X need to have the same number of 'docs'!");

  // Initialise qY randomly and qZ to ones
  qY.resize(J);
  qZ.resize(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    ArrayXXd randm = (ArrayXXd::Random(X[j].size(), maxT)).abs();
    ArrayXd norm = randm.rowwise().sum();
    qY[j] = (randm.log().colwise() - norm.log()).exp();

    qZ[j].resize(X[j].size());

    for (unsigned int i = 0; i < X[j].size(); ++i)
      qZ[j][i].setOnes(X[j][i].rows(), 1);
  }

  bool emptyclasses = true, split = true;
  double F = 0;
  vector<int> stally;

  // Main loop
  while ((split == true) || (emptyclasses == true))
  {

    F = vbem<WJ,WT,CT,CK>(W, X, qY, qZ, weights_j, weights_t, clusters_t,
                          clusters_k, prior_t, prior_k, -1, verbose);

    if (verbose == true)
      cout << '<' << flush; // Notify start bottom-level cluster search

    if (split == false)     // Remove any empty weights
      emptyclasses = prune_clusters_t<WT, CT>(qY, weights_t, clusters_t, 
                                              verbose);
    else
      split = ssplit<WJ,WT,CT,CK>(W, X, clusters_t, clusters_k, qY, qZ, stally,
                                  F, maxK, verbose);

    if (verbose == true)
      cout << '>' << endl;  // Notify end bottom-level cluster search
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of top level clusters = " << clusters_t.size();
    cout << ", and bottom level clusters = " << clusters_k.size() << endl;
    cout << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//

double libcluster::learnMCM (
    const vMatrixXd& W,
    const vvMatrixXd& X,
    vMatrixXd& qY,
    vvMatrixXd& qZ,
    vector<GDirichlet>& weights_j,
    vector<Dirichlet>& weights_t,
    vector<GaussWish>& clusters_t,
    vector<GaussWish>& clusters_k,
    const double prior_t,
    const double prior_k,
    const unsigned int maxT,
    const int maxK,
    const bool verbose,
    const unsigned int nthreads
    )
{

  if (verbose == true)
    cout << "Learning MCM..." << endl;

  // Model selection and Variational Bayes learning
  double F = mcluster<GDirichlet, Dirichlet, GaussWish, GaussWish>(W, X, qY, qZ,
                weights_j, weights_t, clusters_t, clusters_k, prior_t, prior_k,
                maxT, maxK, verbose, nthreads);

  return F;
}
