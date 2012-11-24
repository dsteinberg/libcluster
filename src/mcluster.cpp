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
template <class IW, class SW, class IC> double vbeY (
    const MatrixXd& Oj,          // Image observations for group j
    const vMatrixXd& qZj,        // Cluster assignments for group j
    const IW& weightsj,          // Group image cluster weights
    const vector<SW>& sweights,  // Segment cluster proportions per image clust
    const vector<IC>& iclusters, // Image cluster parameters
    MatrixXd& qYj                // Image cluster assignments for group j
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
    logqYj.col(t)   = qZjiLike.col(t) + E_logwj(t)
                      + iclusters[t].Eloglike(Oj).array();
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
template <class SW, class SC> double vbeZ (
    const MatrixXd& Xji,         // Observations in image i in group j
    const RowVectorXd& qYji,     // Image cluster assignment of this image
    const vector<SW>& sweights,  // Image cluster parameters
    const vector<SC>& sclusters, // Segment cluster parameters
    MatrixXd& qZji               // Observation to cluster assignments
    )
{
  const int K   = sclusters.size(),
            Nji = Xji.rows(),
            T   = sweights.size();

  // Make image cluster global weights from weighted label parameters
  RowVectorXd E_logqYljt = RowVectorXd::Zero(K);

  for (int t = 0; t < T; ++t)
    E_logqYljt.noalias() += qYji(t) * sweights[t].Elogweight().matrix();

  // Find Expectations of log joint observation probs
  MatrixXd logqZji = MatrixXd::Zero(Nji, K);

  for (int k = 0; k < K; ++k)
    logqZji.col(k) = E_logqYljt(k) + sclusters[k].Eloglike(Xji).array();

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
template <class IW, class SW, class IC, class SC> double fenergy (
    const vector<IW>& iweights,  // Group image cluster weights
    const vector<SW>& sweights,  // Image cluster segment proportions
    const vector<IC>& iclusters, // Image cluster parameters
    const vector<SC>& sclusters, // Segment cluster parameters
    const double Fyz,           // Free energy Y terms
    const double Fz              // Free energy Z terms
    )
{
  const int T = sweights.size(),
            K = sclusters.size(),
            J = iweights.size();

  // Class parameter free energy
  double Ft = 0;
  for (int t = 0; t < T; ++t)
  {
    Ft += sweights[t].fenergy() + iclusters[t].fenergy();
  }

  // Cluster parameter free energy
  double Fk = 0;
  for (int k = 0; k < K; ++k)
    Fk += sclusters[k].fenergy();

  // Weight parameter free energy
  double Fw = 0;
  for (int j = 0; j < J; ++j)
    Fw += iweights[j].fenergy();

  return Fw + Ft + Fk + Fyz + Fz;
}


/* Variational Bayes EM for all image mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: the segment cluster indicators, qZ
 *  mutable: the image cluster indicators, qY
 *  mutable: model parameters iweights, sweights, sclusters, iclusters
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
template<class IW, class SW, class IC, class SC> double vbem (
    const vMatrixXd& W,           // Image observations
    const vvMatrixXd& X,          // Segment observations
    vMatrixXd& qY,                // Image labels
    vvMatrixXd& qZ,               // Cluster labels
    vector<IW>& iweights,         // Group image cluster weights
    vector<SW>& sweights,         // Image segment weights
    vector<IC>& iclusters,        // Image cluster parameters
    vector<SC>& sclusters,        // Segment cluster parameters
    const double iclusterprior,   // Image cluster prior
    const double sclusterprior,   // Segment cluster prior
    const int maxit = -1,         // Max VBEM iterations (-1 = no max, default)
    const bool verbose = false    // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = qZ[0][0].cols(),
                     T = qY[0].cols();

  // Construct (empty) parameters
  iweights.resize(J, IW());
  sweights.resize(T, SW());
  iclusters.resize(T, IC(iclusterprior, W[0].cols()));
  sclusters.resize(K, SC(sclusterprior, X[0][0].cols()));

  // Other loop variables for initialisation
  int it = 0;
  double F = numeric_limits<double>::max(), Fold;

  do
  {
    Fold = F;

    // Clear Sufficient Stats
    MatrixXd Ntk = MatrixXd::Zero(T, K); // seg cluster per image cluster count

    for (unsigned int k = 0; k < K; ++k)
      sclusters[k].clearobs();

    for (unsigned int t = 0; t < T; ++t)
      iclusters[t].clearobs();

    // Accumulate sufficient stats from segments and images
    for (unsigned int j = 0; j < J; ++j)
    {
      // Image clusters
      for (unsigned int t = 0; t < T; ++t)
        iclusters[t].addobs(qY[j].col(t), W[j]);

      // Segment clusters and counts per images cluster
      for(unsigned int i = 0; i < X[j].size(); ++i)
      {
        Ntk.noalias() += qY[j].row(i).transpose() * qZ[j][i].colwise().sum();
        for (unsigned int k = 0; k < K; ++k)
          sclusters[k].addobs(qZ[j][i].col(k), X[j][i]);
      }
    }

    // VBM for image cluster weights
    for (unsigned int j = 0; j < J; ++j)
      iweights[j].update(qY[j].colwise().sum());

    // VBM for image cluster parameters and proportions
//    #pragma omp parallel for schedule(guided)
    for (unsigned int t = 0; t < T; ++t)
    {
      sweights[t].update(Ntk.row(t));  // Segment cluster count multinomials.
      iclusters[t].update();           // Gaussian image observations
    }

    // VBM for segment cluster parameters
//    #pragma omp parallel for schedule(guided)
    for (unsigned int k = 0; k < K; ++k)
      sclusters[k].update();

    // Free energy data fit term accumulators
    double Fz = 0, Fyz = 0;

    // VBE for image cluster indicators
    for (unsigned int j = 0; j < J; ++j)
      Fyz += vbeY<IW,SW,IC>(W[j], qZ[j], iweights[j], sweights, iclusters,
                            qY[j]);

    // VBE for segment cluster indicators
    for (unsigned int j = 0; j < J; ++j)
    {
//      #pragma omp parallel for schedule(guided) reduction(+ : Fz)
      for (unsigned int i = 0; i < X[j].size(); ++i)
        Fz += vbeZ<SW,SC>(X[j][i], qY[j].row(i), sweights, sclusters, qZ[j][i]);
    }

    // Calculate free energy of model
    F = fenergy<IW,SW,IC,SC>(iweights, sweights, iclusters, sclusters, Fyz, Fz);

//    cout << "F:" << F << ", Fy:" << Fy << ", Fz:" << Fz << endl;

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

/*  Search in a greedy fashion for an image cluster split that lowers model free
 *    energy, or return false. An attempt is made at looking for good, untried,
 *    split candidates first, as soon as a split canditate is found that lowers
 *    model F, it is returned. This may not be the "best" split, but it is
 *    certainly faster than an exhaustive search for the "best" split.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qY is augmented with a new split if one is found, otherwise left
 *    mutable: qZ is updated split if one is found, otherwise left
 *    mutable tally is a tally time a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
template <class IW, class SW, class IC, class SC> bool isplit (
    const vMatrixXd& W,          // Image Observations
    const vvMatrixXd& X,         // Segment observations
    const vector<IW>& iweights,  // Group weight distributions
    const vector<SW>& sweights,  // Group weight distributions
    const vector<IC>& iclusters, // Image cluster distributions
    const vector<SC>& sclusters, // Segment cluster distributions
    vMatrixXd& qY,               // Probabilities qY
    vvMatrixXd& qZ,              // Probabilities qY
    vector<int>& tally,          // Count of unsuccessful splits
    const double F,              // Current model free energy
    const bool verbose           // Verbose output
    )
{
  const unsigned int J = W.size(),
                     T = iclusters.size();

  // Split order chooser and cluster parameters
  tally.resize(T, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(T);

  // Get cluster parameters and their free energy
//  #pragma omp parallel for schedule(guided)
  for (unsigned int t = 0; t < T; ++t)
  {
    ord[t].k     = t;
    ord[t].tally = tally[t];
    ord[t].Fk    = iclusters[t].fenergy() + sweights[t].fenergy();
  }

  // Get cluster likelihoods
//  #pragma omp parallel for schedule(guided)
  for (unsigned int j = 0; j < J; ++j)
  {
    // Get cluster weights
    ArrayXd logpi = iweights[j].Elogweight();

    // Add in cluster log-likelihood, weighted by responsability
    for (unsigned int t = 0; t < T; ++t)
    {
      double LL = qY[j].col(t).dot((logpi(t)
                               + iclusters[t].Eloglike(W[j]).array()).matrix());

//      #pragma omp atomic
      ord[t].Fk -= LL;
    }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector<ArrayXi> mapidx(J, ArrayXi());
  vMatrixXd qYref(J, MatrixXd()), qYaug(J,MatrixXd()), Ot(J,MatrixXd());
  vvMatrixXd Xt(J), qZt(J);

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator i = ord.begin(); i < ord.end(); ++i)
  {
    const int t = i->k;

    ++tally[t]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (iclusters[t].getN() < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

//    #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
    for (unsigned int j = 0; j < J; ++j)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      ArrayXb partind = (qY[j].col(t).array()>0.5);
      mapidx[j] = partobs(W[j], partind, Ot[j]);  // Copy :-(
      partvvobs(X[j], partind, Xt[j]);            // Copy :-(
      partvvobs(qZ[j], partind, qZt[j]);          // Copy :-(
      Mtot += Ot[j].rows();

      // Initial cluster split
      ArrayXb splitt = iclusters[t].splitobs(Ot[j]);
      qYref[j].setZero(Ot[j].rows(), 2);
      qYref[j].col(0) = (splitt == true).cast<double>();  // Init qZ for split
      qYref[j].col(1) = (splitt == false).cast<double>();

      // keep a track of number of splits
      scount += splitt.count();
    }

    // Don't waste time with clusters that haven't been split sufficiently
    if ( (scount < 2) || (scount > (Mtot-2)) )
      continue;

    // Refine the split
    vector<IW> iwspl;
    vector<IC> icspl;
    vector<SW> swspl;
    vector<SC> scspl;
    vbem<IW,SW,IC,SC>(Ot, Xt, qYref, qZt, iwspl, swspl, icspl, scspl,
                   iclusters[0].getprior(), sclusters[0].getprior(), SPLITITER);

    if (anyempty<IC>(icspl) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
//    #pragma omp parallel for schedule(guided)
    for (unsigned int j = 0; j < J; ++j)
      qYaug[j] = auglabels(t, mapidx[j], (qYref[j].col(1).array()>0.5), qY[j]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    vvMatrixXd qZaug = qZ;      // copy :-(
    double Fsplit = vbem<IW,SW,IC,SC>(W, X, qYaug, qZaug, iwspl, swspl, icspl,
            scspl, iclusters[0].getprior(), sclusters[0].getprior(), SPLITITER);

    if (anyempty<IC>(icspl) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > CONVERGE) )
    {
      qY = qYaug;
      qZ = qZaug;
      tally[t] = 0;   // Reset tally if successfully split
      return true;
    }
  }

  // Failed to find splits
  return false;
}


/*  Search in a greedy fashion for an segment cluster split that lowers model
 *    free energy, or return false. An attempt is made at looking for good,
 *    untried, split candidates first, as soon as a split canditate is found
 *    that lowers model F, it is returned. This may not be the "best" split, but
 *    it is certainly faster than an exhaustive search for the "best" split.
 *
 *    returns: true if a split was found, false if no splits can be found
 *    mutable: qZ is augmented with a new split if one is found, otherwise left
 *    mutable tally is a tally time a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
template <class IW, class SW, class IC, class SC> bool ssplit (
    const vMatrixXd& W,             // Image observations
    const vvMatrixXd& X,            // Segment observations
    const vector<IC>& iclusters,    // Segment cluster Distributions
    const vector<SC>& sclusters,    // Segment cluster Distributions
    vMatrixXd& qY,                  // Image cluster Probabilities qY
    vvMatrixXd& qZ,                 // Segment Cluster Probabilities qZ
    vector<int>& tally,             // Count of unsuccessful splits
    const double F,                 // Current model free energy
    const bool verbose              // Verbose output
    )
{
  const unsigned int J = X.size(),
                     K = sclusters.size();

  // Split order chooser and segment cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);

  // Get cluster parameters and their free energy
  for (unsigned int k = 0; k < K; ++k)
  {
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = sclusters[k].fenergy();
  }

  // Get segment cluster likelihoods
  for (unsigned int j = 0; j < J; ++j)
  {
    // Add in cluster log-likelihood, weighted by global responsability
//    #pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < X[j].size(); ++i)
      for (unsigned int k = 0; k < K; ++k)
      {
        double LL = qZ[j][i].col(k).dot(sclusters[k].Eloglike(X[j][i]));

//        #pragma omp atomic
        ord[k].Fk -= LL;
      }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  vector< vector<ArrayXi> > mapidx(J); // TODO: DOES THIS NEED TO BE A vv???
//  vMatrixXd qYref(J);
  vvMatrixXd qZref(J), qZaug(J), Xk(J);

  // Loop through each potential cluster in order and split it
  for (vector<GreedOrder>::iterator ko = ord.begin(); ko < ord.end(); ++ko)
  {
    const int k = ko->k;

    ++tally[k]; // increase this cluster's unsuccessful split tally by default

    // Don't waste time with clusters that can't really be split min (2:2)
    if (sclusters[k].getN() < 4)
      continue;

    // Now split observations and qZ.
    int scount = 0, Mtot = 0;

    for (unsigned int j = 0; j < J; ++j)
    {
      mapidx[j].resize(X[j].size());
      qZref[j].resize(X[j].size());
      qZaug[j].resize(X[j].size());
      Xk[j].resize(X[j].size());
//      qYref[j].setOnes(X[j].size(), 1);

//      #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
      for (unsigned int i = 0; i < X[j].size(); ++i)
      {
        // Make COPY of the observations with only relevant data points, p > 0.5
        mapidx[j][i] = partobs(X[j][i], (qZ[j][i].col(k).array()>0.5), Xk[j][i]);
        Mtot += Xk[j][i].rows();

        // Initial cluster split
        ArrayXb splitk = sclusters[k].splitobs(Xk[j][i]);
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
    vector<IW> iwspl;
    vector<IC> icspl;
    vector<SW> swspl;
    vector<SC> scspl;
    vMatrixXd qYaug = qY;                             // Copy :-(
    vbem<IW,SW,IC,SC>(W, Xk, qYaug, qZref, iwspl, swspl, icspl, scspl,
                   iclusters[0].getprior(), sclusters[0].getprior(), SPLITITER);

    if (anyempty<SC>(scspl) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
    for (unsigned int j = 0; j < J; ++j)
    {
//      #pragma omp parallel for schedule(guided)
      for (unsigned int i = 0; i < X[j].size(); ++i)
        qZaug[j][i] = auglabels(k, mapidx[j][i],
                                (qZref[j][i].col(1).array() > 0.5), qZ[j][i]);
    }

    // Calculate free energy of this split with ALL data (and refine a bit)
    qYaug = qY;                             // Copy :-(
    double Fs = vbem<IW,SW,IC,SC>(W, X, qYaug, qZaug, iwspl, swspl, icspl,
                    scspl, iclusters[0].getprior(), sclusters[0].getprior(), 1);

    if (anyempty<SC>(scspl) == true) // One cluster only
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


template<class IW, class SW, class IC, class SC> double mcluster (
    const vMatrixXd& W,           // Image observations
    const vvMatrixXd& X,          // Segment observations
    vMatrixXd& qY,                // Image labels
    vvMatrixXd& qZ,               // Cluster labels
    vector<IW>& iweights,         // Group image cluster weights
    vector<SW>& sweights,         // Image segment weights
    vector<IC>& iclusters,        // Image cluster parameters
    vector<SC>& sclusters,        // Segment cluster parameters
    const double iclusterprior,   // Image cluster prior
    const double sclusterprior,   // Segment cluster prior
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

  // Initialise qY and qZ to ones
  qY.resize(J);
  qZ.resize(J);

  for (unsigned int j = 0; j < J; ++j)
  {
    qY[j].setOnes(X[j].size(), 1);
    qZ[j].resize(X[j].size());

    for (unsigned int i = 0; i < X[j].size(); ++i)
      qZ[j][i].setOnes(X[j][i].rows(), 1);
  }

  bool i_split = true, s_split = true;
  double F = 0;
  vector<int> stally, itally;

  // Main loop
  while ((i_split == true) || (s_split == true))
  {

    F = vbem<IW,SW,IC,SC>(W, X, qY, qZ, iweights, sweights, iclusters,
                          sclusters, iclusterprior, sclusterprior, -1, verbose);

    // Start model search heuristics
    if (i_split == true)        // Image clusters
    {
      if (verbose == true)
        cout << '<' << flush;   // Notify start image cluster search

      i_split = isplit<IW,SW,IC,SC>(W, X, iweights, sweights, iclusters,
                                    sclusters, qY, qZ, itally, F, verbose);

      if (verbose == true)
        cout << '>' << endl;   // Notify end image cluster search
    }
    else                        // Segment clusters
    {
      if (verbose == true)
        cout << '(' << flush;   // Notify start segment cluster search

      s_split = ssplit<IW,SW,IC,SC>(W, X, iclusters, sclusters, qY, qZ, stally,
                                    F, verbose);

      if (verbose == true)
        cout << ')' << endl;    // Notify end segment cluster search
    }
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of top level clusters = " << iclusters.size();
    cout << ", and bottom level clusters = " << sclusters.size() << endl;
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
    vector<GDirichlet>& iweights,
    vector<Dirichlet>& sweights,
    vector<GaussWish>& iclusters,
    vector<GaussWish>& sclusters,
    const double iclusterprior,
    const double sclusterprior,
    const bool verbose,
    const unsigned int nthreads
    )
{

  if (verbose == true)
    cout << "Learning MCM..." << endl;

  // Model selection and Variational Bayes learning
  double F = mcluster<GDirichlet, Dirichlet, GaussWish, GaussWish>(W, X, qY, qZ,
                iweights, sweights, iclusters, sclusters, iclusterprior,
                sclusterprior, verbose, nthreads);

  return F;
}
