/*
 * libcluster -- A collection of Bayesian clustering algorithms
 * Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// TODO:
//  - Make a sparse flag for the clusters and weights?
//  - Parallelise better

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
 *  mutable: Image cluster assignment probabilities, qYj
 *  returns: The complete-data free energy, Y and Y+Z dep. terms, for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class IW, class SW> double vbeY (
    const vMatrixXd& qZj,       // Cluster assignments for group j
    const IW& weightsj,         // Group image cluster weights
    const vector<SW>& sweights, // Image cluster parameters
    MatrixXd& qYj               // Image cluster assignments for group j
    )
{
  const unsigned int T  = sweights.size(),
                     Ij = qZj.size(),
                     K  = qZj[0].cols();

  // Get log marginal weight likelihoods
  const ArrayXd E_logwj = weightsj.Elogweight();

  MatrixXd Njik(Ij, K), logqYj(Ij, T);
  ArrayXXd qZjiLike(Ij, T);

  // Get segment cluster counts per image
  for (unsigned int i = 0; i < Ij; ++i)
    Njik.row(i) = qZj[i].colwise().sum();

  // Find Expectations of log joint observation probs
  for (unsigned int t = 0; t < T; ++t)
  {
    qZjiLike.col(t) = Njik * sweights[t].Elogweight().matrix();
    logqYj.col(t)   = E_logwj(t) + qZjiLike.col(t);
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZyj = logsumexp(logqYj);

  // Normalise and Compute Responsibilities
  qYj = (logqYj.colwise() - logZyj).array().exp().matrix();

  return ((qYj.array() * qZjiLike).rowwise().sum() - logZyj.array()).sum();
}


/* The Variational Bayes Expectation step for clusters in each image.
 *
 *  mutable: Segment cluster assignment probabilities, qZji
 *  returns: The complete-data  free energy, Z dep. terms, for group j.
 *  throws: invalid_argument rethrown from other functions.
 */
template <class SW, class C> double vbeZ (
    const MatrixXd& Xji,        // Observations in image i in group j
    const RowVectorXd& qYji,    // Image cluster assignment of this image
    const vector<SW>& sweights, // Image cluster parameters
    const vector<C>& clusters,  // Segment cluster parameters
    MatrixXd& qZji              // Observation to cluster assignments
    )
{
  const int K   = clusters.size(),
            Nji = Xji.rows(),
            T   = sweights.size();

  // Make image cluster global weights from weighted label parameters
  RowVectorXd E_logqYljt = RowVectorXd::Zero(K);

  for (int t = 0; t < T; ++t)
    E_logqYljt.noalias() += qYji(t) * sweights[t].Elogweight().matrix();

  // Find Expectations of log joint observation probs
  MatrixXd logqZji = MatrixXd::Zero(Nji, K);

  for (int k = 0; k < K; ++k)
    logqZji.col(k) = E_logqYljt(k) + clusters[k].Eloglike(Xji).array();

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
template <class IW, class SW, class C> double fenergy (
    const vector<IW>& iweights, // Group image cluster weights
    const vector<SW>& sweights, // Image cluster parameters
    const vector<C>& clusters,  // Segment cluster parameters
    const double Fyz,           // Free energy Y and Z+Y terms
    const double Fz             // Free energy Z terms
    )
{
  const int T = sweights.size(),
            K = clusters.size(),
            J = iweights.size();

  // Class parameter free energy
  double Fc = 0;
  for (int t = 0; t < T; ++t)
    Fc += sweights[t].fenergy();

  // Cluster parameter free energy
  double Fk = 0;
  for (int k = 0; k < K; ++k)
    Fk += clusters[k].fenergy();

  // Weight parameter free energy
  double Fw = 0;
  for (int j = 0; j < J; ++j)
    Fw += iweights[j].fenergy();

  return Fw + Fc + Fk + Fyz + Fz;
}


/* Variational Bayes EM for all image mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: the segment cluster indicators, qZ
 *  mutable: the image cluster indicators, qY
 *  mutable: model parameters iweights, sweights, clusters
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
template <class IW, class SW, class C> double vbem (
    const vvMatrixXd& X,        // Observations JxIjx[NjixD]
    vvMatrixXd& qZ,             // Observations to cluster assigns JxIjx[NjixK]
    vMatrixXd& qY,              // Indicator to label assignments Jx[IjxT]
    vector<IW>& iweights,       // Group weight distributions
    vector<SW>& sweights,       // Image cluster distributions
    vector<C>& clusters,        // Segment cluster Distributions
    const double iclusterprior, // Prior value for image cluster distributions
    const double sclusterprior, // Prior value for segment cluster distributions
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool verbose = false  // Verbose output (default false)
    )
{
  const unsigned int J = X.size(),
                     K = qZ[0][0].cols(),
                     T = qY[0].cols();

  // Construct (empty) parameters
  iweights.resize(J, IW());
  sweights.resize(T, SW(iclusterprior));
  clusters.resize(K, C(sclusterprior, X[0][0].cols()));

  // Other loop variables for initialisation
  int it = 0;
  double F = numeric_limits<double>::max(), Fold;

  do
  {
    Fold = F;

    MatrixXd Ntk = MatrixXd::Zero(T, K);  // Clear Sufficient Stats

    // VBM for image cluster weights
    #pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < J; ++j)
    {
      for(unsigned int i = 0; i < X[j].size(); ++i)
      {
        MatrixXd Ntkji = qY[j].row(i).transpose() * qZ[j][i].colwise().sum();
        #pragma omp critical
        Ntk += Ntkji;
      }

      iweights[j].update(qY[j].colwise().sum());
    }

    // VBM for image cluster parameters
    #pragma omp parallel for schedule(guided)
    for (unsigned int t = 0; t < T; ++t)
      sweights[t].update(Ntk.row(t));  // Weighted multinomials.

    // VBM for segment cluster parameters
    #pragma omp parallel for schedule(guided)
    for (unsigned int k = 0; k < K; ++k)
    {
      clusters[k].clearobs();

      for (unsigned int j = 0; j < J; ++j)
        for(unsigned int i = 0; i < X[j].size(); ++i)
          clusters[k].addobs(qZ[j][i].col(k), X[j][i]);

      clusters[k].update();
    }

    double Fz = 0, Fyz = 0;

    // VBE for image cluster indicators
    #pragma omp parallel for schedule(guided) reduction(+ : Fyz)
    for (unsigned int j = 0; j < J; ++j)
      Fyz += vbeY<IW,SW>(qZ[j], iweights[j], sweights, qY[j]);

    // VBE for segment cluster indicators
    for (unsigned int j = 0; j < J; ++j)
    {
      #pragma omp parallel for schedule(guided) reduction(+ : Fz)
      for (unsigned int i = 0; i < X[j].size(); ++i)
        Fz += vbeZ<SW,C>(X[j][i], qY[j].row(i), sweights, clusters, qZ[j][i]);
    }

    // Calculate free energy of model
    F = fenergy<IW,SW,C>(iweights, sweights, clusters, Fyz, Fz);

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
template <class IW, class SW, class C> bool split_gr (
    const vvMatrixXd& X,            // Observations
    const vector<C>& clusters,      // Cluster Distributions
    const double iclusterprior,     // Prior value for image clusters
    vMatrixXd& qY,                  // Image cluster Probabilities qY
    vvMatrixXd& qZ,                 // Segment Cluster Probabilities qZ
    vector<int>& tally,             // Count of unsuccessful splits
    const double F,                 // Current model free energy
    const bool verbose              // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = clusters.size();

  // Split order chooser and segment cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);

  // Get cluster parameters and their free energy
  for (unsigned int k = 0; k < K; ++k)
  {
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = clusters[k].fenergy();
  }

  // Get segment cluster likelihoods
  for (unsigned int j = 0; j < J; ++j)
  {
    // Add in cluster log-likelihood, weighted by global responsability
    #pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < X[j].size(); ++i)
      for (unsigned int k = 0; k < K; ++k)
      {
        double LL = qZ[j][i].col(k).dot(clusters[k].Eloglike(X[j][i]));

        #pragma omp atomic
        ord[k].Fk -= LL;
      }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector< vector<ArrayXi> > mapidx(J); // TODO: DOES THIS NEED TO BE A vv???
  vMatrixXd qYref(J);
  vvMatrixXd qZref(J), qZaug(J), Xk(J);

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator ko = ord.begin(); ko < ord.end(); ++ko)
  {
    const int k = ko->k;

    ++tally[k]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (clusters[k].getN() < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    for (unsigned int j = 0; j < J; ++j)
    {
      mapidx[j].resize(X[j].size());
      qZref[j].resize(X[j].size());
      qZaug[j].resize(X[j].size());
      Xk[j].resize(X[j].size());
      qYref[j].setOnes(X[j].size(), 1);

      #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
      for (unsigned int i = 0; i < X[j].size(); ++i)
      {
        // Make COPY of the observations with only relevant data points, p > 0.5
        mapidx[j][i] = partobs(X[j][i], (qZ[j][i].col(k).array()>0.5), Xk[j][i]);
        Mtot += Xk[j][i].rows();

        // Initial cluster split
        ArrayXb splitk = clusters[k].splitobs(Xk[j][i]);
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
    vector<IW> wspl;
    vector<SW> lspl;
    vector<C> cspl;
    vbem<IW,SW,C>(Xk, qZref, qYref, wspl, lspl, cspl, iclusterprior,
                  clusters[0].getprior(), SPLITITER);

    if (anyempty<C>(cspl) == true) // One cluster only
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
    vMatrixXd qYaug = qY;                             // Copy :-(
    double Fs = vbem<IW,SW,C>(X, qZaug, qYaug, wspl, lspl, cspl, iclusterprior,
                              clusters[0].getprior(), 1);

    if (anyempty<C>(cspl) == true) // One cluster only
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

/*  Find and remove all empty image clusters.
 *
 *    returns: true if any seights have been deleted, false if all are kept.
 *    mutable: qY may have columns deleted if there are empty weights found.
 *    mutable: sweights if there are empty image clusters found.
 */
template <class SW> bool prune_sweights (
    vMatrixXd& qY,        // Probabilities qY
    vector<SW>& sweights, // weights distributions
    bool verbose = false  // print status
    )
{
  const unsigned int T = sweights.size(),
                     J = qY.size();

  // Look for empty clusters
  ArrayXd Nt(T);
  for (unsigned int t = 0; t < T; ++t)
    Nt(t) = sweights[t].getNk().sum();

  // Find location of empty and full clusters
  ArrayXi eidx, fidx;
  arrfind(Nt.array() < ZEROCUTOFF, eidx, fidx);
  const unsigned int nempty = eidx.size();

  // If everything is not empty, return false
  if (nempty == 0)
    return false;

  if (verbose == true)
    cout << '*' << flush;

  // Delete empty cluster suff. stats.
  for (int i = (nempty - 1); i >= 0; --i)
    sweights.erase(sweights.begin() + eidx(i));

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
 *  mutable: qY the probabilistic image cluster assignments
 *  mutable: qZ the probabilistic observation to segment cluster assignments
 *  mutable: the image cluster weights and parameters.
 *  mutable: the segment clusters weights and parameters.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if free energy increases.
 */
template <class IW, class SW, class C> double scluster (
    const vvMatrixXd& X,        // Observations
    vMatrixXd& qY,              // Image cluster assignments
    vvMatrixXd& qZ,             // Observations to cluster assignments
    vector<IW>& iweights,       // Group weight distributions
    vector<SW>& sweights,       // Image cluster distributions
    vector<C>& clusters,        // Segment cluster Distributions
    const unsigned int T,       // Truncation level for number of weights
    const double iclusterprior, // Prior value for image cluster distributions
    const double sclusterprior, // Prior value for segment cluster distributions
    const bool verbose,         // Verbose output
    const unsigned int nthreads // Number of threads for OpenMP to use
    )
{
  if (nthreads < 1)
    throw invalid_argument("Must specify at least one thread for execution!");
  omp_set_num_threads(nthreads);

  const unsigned int J = X.size();
  unsigned int Itot = 0;

  // Randomly initialise qY and initialise qZ to ones
  qY.resize(J);
  qZ.resize(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    const unsigned int Ij = X[j].size();

    ArrayXXd randm = (ArrayXXd::Random(Ij, T)).abs();
    ArrayXd norm = randm.rowwise().sum();
    qY[j] = (randm.log().colwise() - norm.log()).exp();

    qZ[j].resize(Ij);
    for (unsigned int i = 0; i < Ij; ++i)
      qZ[j][i].setOnes(X[j][i].rows(), 1);

    Itot += Ij;
  }

  // Some input argument checking
  if (T > Itot)
    throw invalid_argument("T must be less than the number of images in X!");

  // Initialise free energy and other loop variables
  bool issplit = true, emptyclasses = true;
  double F = 0;
  vector<int> tally;

  // Main loop
  while ((issplit == true) || (emptyclasses == true))
  {
    // Variational Bayes
    F = vbem<IW,SW,C>(X, qZ, qY, iweights, sweights, clusters, iclusterprior,
                      sclusterprior, -1, verbose);

    // Start model search heuristics
    if (verbose == true)
      cout << '<' << flush; // Notify start search

    if (issplit == false)   // Remove any empty weights
      emptyclasses = prune_sweights<SW>(qY, sweights, verbose);
    else                    // Search for best split, augment qZ if found one
      issplit = split_gr<IW,SW,C>(X, clusters, iclusterprior, qY, qZ, tally, F,
                                  verbose);

    if (verbose == true)
      cout << '>' << endl;      // Notify end search
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of top level clusters = " << sweights.size();
    cout << ", and bottom level clusters = " << clusters.size() << endl;
    cout << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//

double libcluster::learnSCM (
    const vvMatrixXd& X,
    vMatrixXd& qY,
    vvMatrixXd& qZ,
    vector<GDirichlet>& iweights,
    vector<Dirichlet>& sweights,
    vector<GaussWish>& clusters,
    const unsigned int T,
    const double iclusterprior,
    const double sclusterprior,
    const bool verbose,
    const unsigned int nthreads
    )
{

  if (verbose == true)
    cout << "Learning SCM..." << endl;

  // Model selection and Variational Bayes learning
  double F = scluster<GDirichlet, Dirichlet, GaussWish>(X, qY, qZ,
                iweights, sweights, clusters, T, iclusterprior, sclusterprior,
                verbose, nthreads);

  return F;
}

