
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
// Private Constants
//

const int VBE_ITER    = 3;  // Number of times to iterate between the VBE steps


//
//  Private Functions
//


/*
 *
 */
template <class W, class L> double vbeY (
    const MatrixXd& Nik,        // [JxK]
    const W& wdists,
    const vector<L>& ldists,
    MatrixXd& qY                // [JxT]
    )
{
  const int T = ldists.size(),
            I = Nik.rows();

  // Get log marginal weight likelihoods
  const ArrayXd E_logY = wdists.Eloglike();

  // Find Expectations of log joint observation probs
  MatrixXd logqY(I, T);

  for (int t=0; t < T; ++t)
    logqY.col(t) = E_logY(t) + (Nik * ldists[t].Eloglike().matrix()).array();

  // Log normalisation constant of log observation likelihoods
  const VectorXd logZy = logsumexp(logqY);

  // Normalise and Compute Responsibilities
  qY = (logqY.colwise() - logZy).array().exp().matrix();

  return -logZy.sum();
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

  for (int t=0; t < T; ++t)
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


template <class W, class L, class C> double fenergy (
    const W& wdists,
    const vector<L>& ldists,
    const vector<C>& cdists,
    const double Fy,
    const vector<double>& Fz
    )
{
  const int T = ldists.size(),
            K = cdists.size(),
            I = Fz.size();

  // Weight parameter free energy
  const double Fw = wdists.fenergy();

  // Class parameter free energy
  double Fl = 0;
  for (int t=0; t < T; ++t)
    Fl += ldists[t].fenergy();

  // Cluster parameter free energy
  double Fc = 0;
  for (int k = 0; k < K; ++k)
    Fc += cdists[k].fenergy();

  // Cluster observation log-likelihood
  double Fzall = 0;
  for (int i=0; i < I; ++i)
    Fzall += Fz[i];

  return Fw + Fl + Fc + Fy + Fzall;
}


/* Variational Bayes EM for all group mixtures.
 *
 *  returns: Free energy of the whole model.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: the group sufficient stats.
 *  mutable: the model sufficient stats.
 *  throws: invalid_argument rethrown from other functions or if cdists.size()
 *          does not match qZ[i].cols().
 *  throws: runtime_error if there is a negative free energy.
 */
template <class W, class L, class C> double vbem (
    const vector<MatrixXd>& X,  // Observations Jx[NjxD]
    vector<MatrixXd>& qZ,       // Observations to cluster assignments Jx[NjxK]
    MatrixXd& qY,               // Indicator to label assignments [JxT]
    W& wdists,                  // Model class weights
    vector<L>& ldists,          // Class parameters
    vector<C>& cdists,          // Cluster parameters
    const int maxit = -1,       // Max VBEM iterations (-1 = no max, default)
    const bool verbose = false  // Verbose output (default false)
    )
{
  const int I = X.size(),
            K = qZ[0].cols(),
            T = qY.cols();

  // Construct the parameters
//  W wdists;
//  vector<L> ldists(T, L());
//  vector<C> cdists(K, C(SS.getprior(), X[0].cols()));

  double F = numeric_limits<double>::max(), Fold, Fy;
  vector<double> Fz(I);
  int it = 0;

  do
  {
    Fold = F;

    // Calculate some sufficient statistics
    MatrixXd Nik = MatrixXd::Zero(I,K);     // Could pre-allocate these b4 loop?
    vector<MatrixXd> SS1(K), SS2(K);
    for (int i=0; i < I; ++i)
    {
      Nik.row(i) = qZ[i].colwise().sum();   // Sum over groups to get multinoms.
                   // TODO: This could be replaced by the SS struct! SSj.getk(k)

      MatrixXd SS1i, SS2i;
      for (int k=0; k < K; ++k)             // Get cluster suff. stats.
      {
        C::makeSS(qZ[i].col(k), X[i], SS1i, SS2i);
        if (i == 0)
        {
          SS1[k] = SS1i;
          SS2[k] = SS2i;
        }
        else
        {
          SS1[k] += SS1i;
          SS2[k] += SS2i;
        }
      }
    }

    // VBM for class weights
    wdists.update(qY.colwise().sum());

    // VBM for class parameters
    for (int t=0; t < T; ++t)
      ldists[t].update(qY.col(t).transpose()*Nik);  // Weighted multinomials.

    // VBM for cluster parameters
    for (int k=0; k < K; ++k)
      cdists[k].update(Nik.col(k).sum(), SS1[k], SS2[k]);

    // VBE iterations
    for (int e=0; e < VBE_ITER; ++e)
    {

      // VBE for class indicators
      Fy = vbeY<W,L>(Nik, wdists, ldists, qY);

      // VBE for cluster indicators
      for (int i = 0; i < I; ++i)
        Fz[i] = vbeZ<L,C>(X[i], qY.row(i), ldists, cdists, qZ[i]);

    }

    // Calculate free energy of model
    F = fenergy<W,L,C>(wdists, ldists, cdists, Fy, Fz);

    // Check bad free energy step
    if ((F-Fold)/abs(Fold) > libcluster::FENGYDEL)
      cout << '(' << (F-Fold)/abs(Fold) << ')';
//      throw runtime_error("Free energy increase!");

    if (verbose == true)              // Notify iteration
      cout << '-' << flush;
  }
  while ( (abs((Fold-F)/Fold) > libcluster::CONVERGE)
          && ( (it++ < maxit) || (maxit < 0) ) );

  return F;
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
template <class W, class L, class C> double modelselect (
    const vector<MatrixXd>& X,   // Observations
    MatrixXd& qY,                // Class assignments
    vector<MatrixXd>& qZ,        // Observations to cluster assignments
    const unsigned int T,        // Truncation level for number of classes
    const bool verbose           // Verbose output
    )
{
  const unsigned int I = X.size();

  if (T > I)
    throw invalid_argument("T must be <= I, the number of groups in X!");

  // Temporary test initialisation
  ArrayXXd randm = (ArrayXXd::Random(I, T)).abs();
  ArrayXd norm = randm.rowwise().sum();
  qY= (randm.log().colwise() - norm.log()).exp();
//  qY.setConstant(I,T,1/T);

//   TEMP
  const int K = qZ[0].cols();

  // Construct the parameters -- TODO: set cdists prior outside this function
  W wdists;
  vector<L> ldists(T, L());
  vector<C> cdists(K, C(libcluster::PRIORVAL, X[0].cols()));

  double F = vbem<W,L,C>(X, qZ, qY, wdists, ldists, cdists, -1, verbose);

  // Print finished notification if verbose
  if (verbose == true)
  {
    cout << "Finished!" << endl;
    cout << "Number of clusters = " << K << endl;
    cout << "Free energy = " << F << endl;

    cout << endl << "Model weights:" << endl;
    cout << wdists.Eloglike().transpose().exp() << endl;

    cout << endl << "Class parameters:" << endl;
    for (unsigned int t=0; t < T; ++t)
      cout << ldists[t].Eloglike().transpose().exp() << " | Nt = "
           << qY.col(t).sum() << endl;

    cout << endl << "Cluster weights:" << endl;
    for (unsigned int i=0; i < I; ++i)
      cout << qZ[i].colwise().sum() / X[i].rows() << endl;
  }

  return F;
}


//
// Public Functions
//


double libcluster::learnTOP (
    const vector<MatrixXd>& X,
    MatrixXd& qY,
    vector<MatrixXd>& qZ,
    const unsigned int T,
    const bool verbose
    )
{
  const int I = X.size();

  // TEMP
  const int K = 3;

  // Create qZ
  qZ.resize(I);
  for (int i=0; i < I; ++i)
  {
    ArrayXXd randm = (ArrayXXd::Random(X[i].rows(), K) + 0.1).abs();
    ArrayXd norm = randm.rowwise().sum();
    qZ[i] = (randm.log().colwise() - norm.log()).exp();
  }

  // Model selection and Variational Bayes learning
  return modelselect<Dirichlet, Dirichlet, GaussWish>(X, qY, qZ, T, verbose);

}
