// TODO:
//  - Make a bootstrap function for starting from a previous Suff. stats.
//  - Make a sparse flag for the clusters and classes?
//  - Neaten up the split_gr() function.
//  - Add in an optional split_ex() function.
//  - Parallelise
//  - Find a better way of outputting the class parameters.
//  - is there a way to re-use updateSS() and split() between this and
//    cluster.cpp?

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
// Private Constants
//

const int VBE_ITER = 1;  // Number of times to iterate between the VBE steps

//
//  Private Functions
//


/* Update the document and model sufficient statistics based on assignments qZi.
 *
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 */
template <class C> ArrayXd updateSS (
    const MatrixXd& Xi,         // Observations in document i
    const MatrixXd& qZi,        // Observations to document mixture assignments
    libcluster::SuffStat& SSi,  // Sufficient stats of document i
    libcluster::SuffStat& SS    // Sufficient stats of whole model
    )
{
  const int K = qZi.cols();

  SS.subSS(SSi);                      // get rid of old document SS contribution

  const ArrayXd Nik = qZi.colwise().sum();  // count obs. in this document
  MatrixXd SS1, SS2;                        // Suff. Stats

  // Create Sufficient statistics
  for (int k = 0; k < K; ++k)
  {
    C::makeSS(qZi.col(k), Xi, SS1, SS2);
    SSi.setSS(k, Nik(k), SS1, SS2);
  }

  SS.addSS(SSi); // Add new document SS contribution

  return Nik;
}


/*
 *
 */
template <class W, class L> ArrayXd vbeY (
    const MatrixXd& Nik,        // [IxK] sum_n q(z_in = k)
    const W& wdists,
    const vector<L>& ldists,
    MatrixXd& qY                // [IxT]
    )
{
  const int T = ldists.size(),
            I = Nik.rows();

  // Get log marginal weight likelihoods
  const ArrayXd E_logY = wdists.Eloglike();

  // Find Expectations of log joint observation probs
  MatrixXd logqY(I, T);
  ArrayXXd qZiPi = ArrayXXd::Zero(I,T);

  for (int t = 0; t < T; ++t)
  {
    qZiPi.col(t) = Nik * ldists[t].Eloglike().matrix();
    logqY.col(t) = E_logY(t) + qZiPi.col(t);
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZy = logsumexp(logqY);

  // Normalise and Compute Responsibilities
  qY = (logqY.colwise() - logZy).array().exp().matrix();

  return (qY.array() * qZiPi).rowwise().sum() - logZy.array();
}


/*
 *
 */
template <class L, class C> double vbeZ (
    const MatrixXd& Xi,
    const RowVectorXd& qYi,
    const vector<L>& ldists,
    const vector<C>& cdists,
    MatrixXd& qZi
    )
{
  const int K  = cdists.size(),
            Ni = Xi.rows(),
            T  = ldists.size();

  // Make cluster global weights from weighted label parameters
  ArrayXd E_logZt = ArrayXd::Zero(K);

  for (int t = 0; t < T; ++t)
    E_logZt += qYi(t) * ldists[t].Eloglike();

  // Find Expectations of log joint observation probs
  MatrixXd logqZi(Ni, K);

  for (int k = 0; k < K; ++k)
    logqZi.col(k) = E_logZt(k) + cdists[k].Eloglike(Xi).array();

  // Log normalisation constant of log observation likelihoods
  const VectorXd logZzi = logsumexp(logqZi);

  // Normalise and Compute Responsibilities
  qZi = (logqZi.colwise() - logZzi).array().exp().matrix();

  return -logZzi.sum();
}


/*
 *
 */
template <class W, class L, class C> double fenergy (
    const W& wdists,
    const vector<L>& ldists,
    const vector<C>& cdists,
    const ArrayXd& Fy,
    const ArrayXd& Fz,
    vector<libcluster::SuffStat>& SSi,  // Document sufficient statistics
    libcluster::SuffStat& SS            // Model Sufficient statistics
    )
{
  const int T = ldists.size(),
            K = cdists.size(),
            I = Fz.size();

  // Weight parameter free energy
  const double Fw = wdists.fenergy();

  // Class parameter free energy
  double Fl = 0;
  for (int t = 0; t < T; ++t)
    Fl += ldists[t].fenergy();

  // Cluster parameter free energy
  double Fc = 0;
  for (int k = 0; k < K; ++k)
    Fc += cdists[k].fenergy();

  // Free energy of the documents likelihoods
  for (int i = 0; i < I; ++i)
  {
    SS.subF(SSi[i]);  // Remove old documents F contribution
    SSi[i].setF(Fy(i) + Fz(i));
    SS.addF(SSi[i]);  // Add in the new documents F contribution
  }

  return Fw + Fl + Fc + SS.getF();
}


/* Variational Bayes EM for all document mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument rethrown from other functions or if cdists.size()
 *          does not match qZ[i].cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class L, class C> double vbem (
    const vector<MatrixXd>& X,  // Observations Ix[NjxD]
    vector<MatrixXd>& qZ,       // Observations to cluster assignments Ix[NjxK]
    MatrixXd& qY,               // Indicator to label assignments [IxT]
    vector<libcluster::SuffStat>& SSi, // Sufficient stats of each document
    libcluster::SuffStat& SS,   // Sufficient stats of whole model
    MatrixXd& classparams,      // Document class parameters
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool verbose = false  // Verbose output (default false)
    )
{
  const int I = X.size(),
            K = qZ[0].cols(),
            T = qY.cols();

  // Construct the parameters
  W wdists;
  vector<L> ldists(T, L());
  vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

  double F = numeric_limits<double>::max(), Fold;
  ArrayXd Fz(I), Fy(I);
  MatrixXd Nik(I,K);
  int it = 0;

  do
  {
    Fold = F;

    // VBM for class weights
    wdists.update(qY.colwise().sum());

    // Calculate some sufficient statistics
    for (int i = 0; i < I; ++i)
      Nik.row(i) = updateSS<C>(X[i], qZ[i], SSi[i], SS);

    // VBM for class parameters
    for (int t = 0; t < T; ++t)
      ldists[t].update(qY.col(t).transpose()*Nik);  // Weighted multinomials.

    // VBM for cluster parameters
    for (int k = 0; k < K; ++k)
      cdists[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));

    // VBE iterations, need to iterate between Z and Y for best results
    for (int e = 0; e < VBE_ITER; ++e)
    {
      // VBE for class indicators
      Fy = vbeY<W,L>(Nik, wdists, ldists, qY);

      // VBE for cluster indicators
      for (int i = 0; i < I; ++i)
        Fz(i) = vbeZ<L,C>(X[i], qY.row(i), ldists, cdists, qZ[i]);
    }

    // Calculate free energy of model
    F = fenergy<W,L,C>(wdists, ldists, cdists, Fy, Fz, SSi, SS);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > libcluster::FENGYDEL)
      cout << '(' << (F-Fold)/abs(Fold) << ')';

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > libcluster::CONVERGE)
          && ( (it++ < maxit) || (maxit < 0) ) );

  // Get the document glass parameters
  classparams.setZero(T,K);
  for (int t = 0; t < T; ++t)
    classparams.row(t) = ldists[t].Eloglike().transpose().exp();

  return F;
}


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
template <class W, class L, class C> bool split_gr (
    const vector<MatrixXd>& X,               // Observations
    const vector<libcluster::SuffStat>& SSi, // Sufficient stats of groups
    const libcluster::SuffStat& SS,          // Sufficient stats
    const double F,                          // Current model free energy
    MatrixXd& qY,                            // Class Probabilities qY
    vector<MatrixXd>& qZ,                    // Cluster Probabilities qZ
    vector<int>& tally,                      // Count of unsuccessful splits
    const bool verbose                       // Verbose output
    )
{
  const unsigned int I = X.size(),
                     K = SS.getK();

  // Split order chooser and cluster parameters
  tally.resize(K, 0); // Make sure tally is the right size
  vector<GreedOrder> ord(K);
  vector<C> csplit(K, C(SS.getprior(), X[0].cols()));

  // Get cluster parameters and their free energy
//  #pragma omp parallel for schedule(guided)
  for (unsigned int k = 0; k < K; ++k)
  {
    csplit[k].update(SS.getN_k(k), SS.getSS1(k), SS.getSS2(k));
    ord[k].k     = k;
    ord[k].tally = tally[k];
    ord[k].Fk    = csplit[k].fenergy();
  }

  // Get cluster likelihoods
//  #pragma omp parallel for schedule(guided)
  for (unsigned int i = 0; i < I; ++i)
  {
    // Get GLOBAL cluster weights
    L wsplit;
    wsplit.update(qZ[i].colwise().sum());
    ArrayXd logpi = wsplit.Eloglike();

    // Add in cluster log-likelihood, weighted by responsability
    for (unsigned int k = 0; k < K; ++k)
    {
      double LL = logpi(k) + qZ[i].col(k).dot(csplit[k].Eloglike(X[i]));

//      #pragma omp atomic
      ord[k].Fk -= LL;
    }
  }

  // Sort clusters by split tally, then free energy contributions
  sort(ord.begin(), ord.end(), greedcomp);

  // Pre allocate big objects for loops (this makes a runtime difference)
  MatrixXd classparams;
  vector<ArrayXi> mapidx(I, ArrayXi());
  vector<MatrixXd> qZref(I,MatrixXd()), qZaug(I,MatrixXd()), Xk(I,MatrixXd());

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

//    #pragma omp parallel for schedule(guided) reduction(+ : Mtot, scount)
    for (unsigned int i = 0; i < I; ++i)
    {
      // Make COPY of the observations with only relevant data points, p > 0.5
      mapidx[i] = partX(X[i], (qZ[i].col(k).array()>0.5), Xk[i]);  // Copy :-(
      Mtot += Xk[i].rows();

      // Initial cluster split
      ArrayXb splitk = csplit[k].splitobs(Xk[i]);
      qZref[i].setZero(Xk[i].rows(), 2);
      qZref[i].col(0) = (splitk == true).cast<double>();  // Init qZ for split
      qZref[i].col(1) = (splitk == false).cast<double>();

      // keep a track of number of splits
      scount += splitk.count();
    }

    // Don't waste time with clusters that haven't been split sufficiently
    if ( (scount < 2) || (scount > (Mtot-2)) )
      continue;

    // Refine the split
    MatrixXd qYref = MatrixXd::Ones(I,1);
    libcluster::SuffStat SSref(SS.getprior());
    vector<libcluster::SuffStat> SSiref(I, libcluster::SuffStat(SS.getprior()));
    vbem<W,L,C>(Xk, qZref, qYref, SSiref, SSref, classparams,
                libcluster::SPLITITER);

    if (anyempty(SSref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
//    #pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < I; ++i)
      qZaug[i] = augmentqZ(k, mapidx[i], (qZref[i].col(1).array()>0.5), qZ[i]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    MatrixXd qYaug = qY;                                          // Copy :-(
    libcluster::SuffStat SSaug = SS;                              // Copy :-(
    vector<libcluster::SuffStat> SSiaug = SSi;                    // Copy :-(
    double Fsplit = vbem<W,L,C>(X, qZaug, qYaug, SSiaug, SSaug, classparams, 1);

    if (anyempty(SSaug) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > libcluster::CONVERGE) )
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


/* The model selection algorithm
 *
 *  returns: Free energy of the final model
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if free energy increases.
 */
template <class W, class L, class C> double modelselect (
    const vector<MatrixXd>& X,   // Observations
    MatrixXd& qY,                // Class assignments
    vector<MatrixXd>& qZ,        // Observations to cluster assignments
    vector<libcluster::SuffStat>& SSdocs, // Sufficient stats of documents
    libcluster::SuffStat& SS,    // Sufficient stats
    MatrixXd& classparams,       // Document class parameters
    const unsigned int T,        // Truncation level for number of classes
    const bool verbose           // Verbose output
    )
{
  const unsigned int I = X.size();

  // Some input argument checking
  if (T > I)
    throw invalid_argument("T must be <= I, the number of documents in X!");

  if (SSdocs.size() != X.size())
    throw invalid_argument("SSdocs and X must be the same size!");

  // Randomly initialise qY
  {
    ArrayXXd randm = (ArrayXXd::Random(I, T)).abs();
    ArrayXd norm = randm.rowwise().sum();
    qY = (randm.log().colwise() - norm.log()).exp();
  }
//  qY.setOnes(I,1);

  qZ.resize(I);
  for (unsigned int i = 0; i < I; ++i)
    qZ[i] = MatrixXd::Ones(X[i].rows(), 1);

  // Initialise free energy and other loop variables
  bool   issplit = true;
  double F;
  vector<int> tally;

  // Main loop
  while (issplit == true)
  {
    F = vbem<W,L,C>(X, qZ, qY, SSdocs, SS, classparams, -1, verbose);

    // Start cluster splitting
    if (verbose == true)
      cout << '<' << flush;  // Notify start splitting

    // Search for best split, augment qZ if found one
    issplit = split_gr<W,L,C>(X, SSdocs, SS, F, qY, qZ, tally, verbose);

    if (verbose == true)
      cout << '>' << endl;   // Notify end splitting
  }

  // Hard assign qY
//  for (unsigned int i = 0; i < I; ++i)
//  {
//    int t;
//    qY.row(i).maxCoeff(&t);
//    qY.row(i).setZero();
//    qY(i,t) = 1;
//  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of classes = " << (qY.colwise().sum().array() >= 1).count();
    cout << ", and clusters = " << SS.getK() << endl;
    cout << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//


double libcluster::learnTCM (
    const vector<MatrixXd>& X,
    MatrixXd& qY,
    vector<MatrixXd>& qZ,
    vector<libcluster::SuffStat>& SSdocs,
    libcluster::SuffStat& SS,
    MatrixXd& classparams,
    const unsigned int T,
    const bool verbose
    )
{

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << "TCM..." << endl;

  // Model selection and Variational Bayes learning
  return modelselect<Dirichlet, Dirichlet, NormGamma>(X, qY, qZ, SSdocs, SS,
                                                      classparams, T, verbose);
}
