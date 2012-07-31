// TODO:
//  - Make a bootstrap function for starting from a previous Suff. stats.
//  - Make a sparse flag for the clusters and classes?
//  - Neaten up the split_gr() function.
//  - Add in an optional split_ex() function.
//  - Parallelise
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
using namespace libcluster;


//
//  Variational Bayes Private Functions
//


/* Update the document and model sufficient statistics based on assignments qZi.
 *
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 */
template <class C> ArrayXd updateSS (
    const MatrixXd& Xi,         // Observations in document i
    const MatrixXd& qZi,        // Observations to document mixture assignments
    SuffStat& SSi,  // Sufficient stats of document i
    SuffStat& SS    // Sufficient stats of whole model
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


/* TODO
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
  const ArrayXd E_logY = wdists.Elogweight();

  // Find Expectations of log joint observation probs
  MatrixXd logqY(I, T);
  ArrayXXd qZiPi = ArrayXXd::Zero(I,T);

  for (int t = 0; t < T; ++t)
  {
    qZiPi.col(t) = Nik * ldists[t].Elogweight().matrix();
    logqY.col(t) = E_logY(t) + qZiPi.col(t);
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZy = logsumexp(logqY);

  // Normalise and Compute Responsibilities
  qY = (logqY.colwise() - logZy).array().exp().matrix();

  return (qY.array() * qZiPi).rowwise().sum() - logZy.array();
}


/* TODO
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
    E_logZt += qYi(t) * ldists[t].Elogweight();

  // Find Expectations of log joint observation probs
  MatrixXd logqZi = MatrixXd::Zero(Ni, K);

  for (int k = 0; k < K; ++k)
    logqZi.col(k) = E_logZt(k) + cdists[k].Eloglike(Xi).array();

  // Log normalisation constant of log observation likelihoods
  const VectorXd logZzi = logsumexp(logqZi);

  // Normalise and Compute Responsibilities
  qZi = (logqZi.colwise() - logZzi).array().exp().matrix();

  return -logZzi.sum();
}


/* TODO
 *
 */
template <class W, class L, class C> double fenergy (
    const W& wdists,
    const vector<L>& ldists,
    const vector<C>& cdists,
    const ArrayXd& Fy,
    const ArrayXd& Fz,
    vSuffStat& SSi,         // Document sufficient statistics
    SuffStat& SS            // Model Sufficient statistics
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
 *  mutable: the cluster indicators, qZ
 *  mutable: the class indicators, qY
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class L, class C> double vbem (
    const vMatrixXd& X,  // Observations Ix[NjxD]
    vMatrixXd& qZ,       // Observations to cluster assignments Ix[NjxK]
    MatrixXd& qY,               // Indicator to label assignments [IxT]
    vSuffStat& SSi, // Sufficient stats of each document
    libcluster::SuffStat& SS,   // Sufficient stats of whole model
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

    // VBE for class indicators
    Fy = vbeY<W,L>(Nik, wdists, ldists, qY);

    // VBE for cluster indicators
    for (int i = 0; i < I; ++i)
      Fz(i) = vbeZ<L,C>(X[i], qY.row(i), ldists, cdists, qZ[i]);

    // Calculate free energy of model
    F = fenergy<W,L,C>(wdists, ldists, cdists, Fy, Fz, SSi, SS);

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
 *    mutable tally is a tally time a cluster has been unsuccessfully split
 *    throws: invalid_argument rethrown from other functions
 *    throws: runtime_error from its internal VBEM calls
 */
template <class W, class L, class C> bool split_gr (
    const vMatrixXd& X,             // Observations
    const vSuffStat& SSi,           // Sufficient stats of groups
    const libcluster::SuffStat& SS, // Sufficient stats
    const double F,                 // Current model free energy
    MatrixXd& qY,                   // Class Probabilities qY
    vMatrixXd& qZ,                  // Cluster Probabilities qZ
    vector<int>& tally,             // Count of unsuccessful splits
    const bool verbose              // Verbose output
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
    ArrayXd logpi = wsplit.Elogweight();

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
  vector<ArrayXi> mapidx(I, ArrayXi());
  vMatrixXd qZref(I,MatrixXd()), qZaug(I,MatrixXd()), Xk(I,MatrixXd());

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
    SuffStat SSref(SS.getprior());
    vSuffStat SSiref(I, SuffStat(SS.getprior()));
    vbem<W,L,C>(Xk, qZref, qYref, SSiref, SSref, SPLITITER);

    if (anyempty(SSref) == true) // One cluster only
      continue;

    // Map the refined splits back to original whole-data problem
//    #pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < I; ++i)
      qZaug[i] = augmentqZ(k, mapidx[i], (qZref[i].col(1).array()>0.5), qZ[i]);

    // Calculate free energy of this split with ALL data (and refine a bit)
    MatrixXd qYaug = qY;                                          // Copy :-(
    SuffStat SSaug = SS;                              // Copy :-(
    vSuffStat SSiaug = SSi;                    // Copy :-(
    double Fsplit = vbem<W,L,C>(X, qZaug, qYaug, SSiaug, SSaug, 1);

    if (anyempty(SSaug) == true) // One cluster only
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      cout << '=' << flush;

    // Test whether this cluster split is a keeper
    if ( (Fsplit < F) && (abs((F-Fsplit)/F) > CONVERGE) )
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


/* TODO
 *
 */
template<class L> MatrixXd get_classparams (
    const MatrixXd& qY,
    const vMatrixXd& qZ
    )
{
  const int T = qY.cols(),
            K = qZ[0].cols(),
            I = qZ.size();

  MatrixXd Nik(I, K), classparams(T, K);
  L ldists;

  // Document multinomial counts
  for (int i = 0; i < I; ++i)
    Nik.row(i) = qZ[i].colwise().sum();

  // Create class parameters
  for (int t = 0; t < T; ++t)
  {
    ldists.update(qY.col(t).transpose()*Nik);  // Weighted multinomials.
    classparams.row(t) = ldists.Elogweight().transpose().exp();
  }

  return classparams;
}


/* TODO
 *
 */
bool prune_classes (MatrixXd& qY)
{

  // Find empty classes, count them
  ArrayXi Ypop, Yemp;
  arrfind(qY.colwise().sum().array() > ZEROCUTOFF, Ypop, Yemp);
  const unsigned int Tpop = Ypop.size();

  // No empty classes
  if (Tpop == qY.cols())
    return false;

  // Now copy only populated columns to new qY matrix
  MatrixXd qYnew = MatrixXd::Zero(qY.rows(), Tpop);
  for (unsigned int t = 0; t < Tpop; ++t)
    qYnew.col(t) = qY.col(Ypop(t));

  // Copy back to qY, return true
  qY = qYnew;
  return true;
}


/* The model selection algorithm
 *
 *  returns: Free energy of the final model
 *  mutable: qY the probabilistic document to class assignments
 *  mutable: qZ the probabilistic observation to cluster assignments
 *  mutable: the document sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument from other functions.
 *  throws: runtime_error if free energy increases.
 */
template <class W, class L, class C> double modelselect (
    const vMatrixXd& X,   // Observations
    MatrixXd& qY,                // Class assignments
    vMatrixXd& qZ,        // Observations to cluster assignments
    vSuffStat& SSdocs, // Sufficient stats of documents
    SuffStat& SS,    // Sufficient stats
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

  // TODO
  // bootstrap()

  // Randomly initialise qY
  {
    ArrayXXd randm = (ArrayXXd::Random(I, T)).abs();
    ArrayXd norm = randm.rowwise().sum();
    qY = (randm.log().colwise() - norm.log()).exp();
  }

  // Initialise qZ
  qZ.resize(I);
  for (unsigned int i = 0; i < I; ++i)
    qZ[i] = MatrixXd::Ones(X[i].rows(), 1);

  // Initialise free energy and other loop variables
  bool issplit = true;
  double F = 0;
  vector<int> tally;

  // Main loop
  while (issplit == true)
  {
    // Variational Bayes
    F = vbem<W,L,C>(X, qZ, qY, SSdocs, SS, -1, verbose);

    // Remove any empty clusters
    bool isremk = prune_clusters(qZ, SSdocs, SS);

    if ( (verbose == true) && (isremk == true) )
      cout << 'x' << flush;

    // Start model search heuristics
    if (verbose == true)
      cout << '<' << flush;     // Notify start search

    // Search for best split, augment qZ if found one
    issplit = split_gr<W,L,C>(X, SSdocs, SS, F, qY, qZ, tally, verbose);

    if (verbose == true)
      cout << '>' << endl;     // Notify end search
  }

  // Remove any empty classes
  prune_classes(qY);

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of classes = " << qY.cols();
    cout << ", and clusters = " << SS.getK() << endl;
    cout << "Free energy = " << F << endl;
  }

  return F;
}


//
// Public Functions
//


double libcluster::learnTCM (
    const vMatrixXd& X,
    MatrixXd& qY,
    vMatrixXd& qZ,
    vSuffStat& SSdocs,
    SuffStat& SS,
    MatrixXd& classparams,
    const unsigned int T,
    const bool verbose
    )
{

  // Model selection and Variational Bayes learning
  if (verbose == true)
    cout << "Learning " << "TCM..." << endl;

  // Model selection and Variational Bayes learning
  double F = modelselect<GDirichlet, Dirichlet, GaussWish>(X, qY, qZ, SSdocs,
                                                           SS, T, verbose);

  // Get the document class parameters
  classparams = get_classparams<Dirichlet>(qY, qZ);

  return F;
}

