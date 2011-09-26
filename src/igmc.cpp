// TODO -- maybe
// - Make a sparse version!
// - See if there is a closed form, series limit version of changegroups()
// - Make the calcSS() interface a bit nicer, as well as some of the oher IGMC
//  interfaces.
// - Make the way I handle free energy a bit nicer/neater! Also maybe pass out
//   more detail for the free energy, so it can be monitored
// - Find a nice way to not have to do Njk = qZ.colwise().sum() twice? calcSS()
//   and vbmaximisationj();
// - Handle GWprior() constructor throwing errors!
// - Profile and look for excessive greedy copying like in augmentqZ() with
//   qZsplit, etc in learnIGMC().

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
// Private Types, Globals and Structures
//

// Free energy structure for sotring the current state of free energy
struct Fenergy
{
  Fenergy() : tot(numeric_limits<double>::max()) {}

  double tot;   // Total free energy
  double pij;   // Group's weight (pi) free energy (not discounted)
  double xzj;   // Complete data free energy (not dicounted)
};


//
// Prototypes
//


/* Incremental Varational Bayes EM for a group in the I-GMC.
 *  returns: the number of iterations of VB performed.
 *  mutable: variational posterior approximations to p(Z|X).
 *  mutable: spost, all group posterior parameters.
 *  mutable: gpost, the model gpost parameters.
 *  throws: invalid_argument rethrown from other functions.
 *  throws: runtime_error if there is a negative free energy.
 */
int vbeminc (
    const MatrixXd& X,          // Observations of the current group
    MatrixXd& qZ,               // Group obs. to model mixture assignments
    const libcluster::IGMC& igmc,  // The current I-GMC object
    const SBprior& sprior,      // Model SB priors
    const GWprior& gprior,      // Model GW priors
    vector<SBposterior>& spost, // Current posterior group parameters
    vector<GWposterior>& gpost, // Posterior model parameters
    Fenergy& F,                 // Free energy structure
    const int maxit = -1,       // Max 'meta' iterations (-1 = none, default)
//    const bool sparse = false,  // Do sparse updates to groups (default false)
    const bool verbose = false, // Verbose output (default false)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    );

/* Variational Bayes Maximisation step for the I-GMC group GD parameters.
 *  returns: An array of the descending size order of the group clusters
 *  mutable: spost, the group posterior parameters.
 */
ArrayXi vbmaximisationj (
    const MatrixXd& qZ,         // Observations to model mixture assignments
    const SBprior& sprior,      // Group SB priors
    vector<SBposterior>& spost  // posterior group SB parameters
//    const bool sparse = false    // Do sparse updates to groups (default false)
    );

/* Variational Bayes Maximisation step for the I-GMC model cluster GW params.
 *  returns: An array of the descending size order of the group clusters
 *  mutable: gpost, the model posterior parameters.
 *  throws: invalid_argument if iW is not PSD.
 */
void vbmaximisationk (
    const MatrixXd& X,          // Observations in group j
    const MatrixXd& qZ,         // Observations to model mixture assignments
    const libcluster::IGMC& igmc,  // The current I-GMC object
    const GWprior& gprior,      // Model GW priors
    vector<GWposterior>& gpost  // posterior GW model parameters
//    const bool sparse = false    // Do sparse updates to groups (default false)
    );

/* The Variational Bayes Expectation step for a group of data.
 *  mutable: spost, vector of group posterior variational parameters
 *  mutable: Assignment probabilities, qZ
 *  returns: negative sum of the log normalisation constant, sum_{n}(p(Xj)).
 *           This is also the free energy of the observations, Fzj.
 *  throws: invalid_argument rethrown from other functions.
 *  NOTE: no discount to F is applied.
 */
double vbexpectationj (
    const MatrixXd& X,         // Observations in group J
    MatrixXd& qZ,              // Observations to group mixture assignments
    vector<SBposterior>& spost,       // Posterior SB group parameters
    const vector<GWposterior>& gpost, // Posterior GW model parameters
    const ArrayXi& order       // Descending size order of the group clusters
//    const bool sparse = false   // Do sparse updates to groups (default false)
    );

/* Calculates the free energy for terms that factor over groups, j.
 *  returns: the free energy of the parameter distribution, pi_j, over j.
 *  NOTE: no discount to F is applied.
 */
double fenergySBj (
    const SBprior& sprior,             // Group SB priors
    const vector<SBposterior>& spost,  // Posterior SB group parameters
    const ArrayXi& order               // Cluster weight order.
    );

/* Calculates the free energy for terms that factor over model clusters, k.
 *  returns: the free energy of the parameter distributions over k.
 */
double fenergyGW (
    const GWprior& gprior,           // Model GW priors
    const vector<GWposterior>& gpost // Posterior GW model parameters
    );

/* Split all of the mixtures.
 *  returns: the free energy of the best split
 *  mutable: qZ, the observation assignment probabilites
 *  throws: invalid_argument rethrown from other functions
 *  throws: runtime_error from its internal VBEM calls
 */
double splitinc (
    const MatrixXd& X,                // Observations
    const MatrixXd& qZ,               // Assignment probs
    const libcluster::IGMC& igmc,        // The current I-GMC object
    const SBprior& sprior,            // Prior hyperparameter values
    const GWprior& gprior,            // Prior hyperparameter value
    const vector<GWposterior>& gpost, // Posterior Hyperparameter values
    MatrixXd& qZsplit,                // Assignment probs of best split
//    const bool sparse = false       // Do sparse updates (default false)
    const bool verbose = false, // Verbose output (default false)
    ostream& ostrm = cout       // Stream to print notification (cout default)
    );

/*
 * TODO
 */
int remempty (
    MatrixXd& qZ,
    libcluster::IGMC& igmc,
    vector<SBposterior>& spost,
    vector<GWposterior>& gpost
  );


//
// Public Member Functions
//

libcluster::IGMC::IGMC (
    unsigned int J,
    unsigned int D,
    double kappa,
    double tau0,
    const RowVectorXd& cmeanp,
    double cwidthp
    )
    : tau(1), lambda(0), rho(1), Fpi(0), Fxz(0)
{
  if (kappa < 0)
    throw invalid_argument("kappa must be greater than or equal to 0!");
  if (tau0 < 1)
    throw invalid_argument("tau0 must be greater than or equal to 1!");
  if (J < 1)
    throw invalid_argument("J must be greater than or equal to 1!");
  if (D < 1)
    throw invalid_argument("D must be greater than or equal to 1!");
  if (cwidthp <= 0)
    throw invalid_argument("cwidthp must be greater than 0!");
  if(cmeanp.cols() != D)
    throw invalid_argument("cmeanp must have a length of D!");

  this->J = J;
  this->D = D;
  this->K = 0;
  this->kappa = kappa;
  this->tau0 = tau0;
  this->cwidthp = cwidthp;
  this->cmeanp = cmeanp;
}


libcluster::IGMC::IGMC (const IGMC& igmc, unsigned int k)
  : J(igmc.J), tau(igmc.tau), kappa(igmc.kappa), tau0(igmc.tau0),
    lambda(igmc.lambda), rho(igmc.rho), cmeanp(igmc.cmeanp),
    cwidthp(igmc.cwidthp), Fpi(0), Fxz(0)
{
  this->D = igmc.D;

  // A few checks
  if ((igmc.K == 0) || (k >= igmc.K))
    return;     // There are no stored suff. stats. yet, so just return

  this->K = 1;
  this->Nk_.push_back(igmc.Nk_[k]);
  this->Xk_.push_back(igmc.Xk_[k]);
  this->Rk_.push_back(igmc.Rk_[k]);
}


void libcluster::IGMC::setcounts (unsigned int J, unsigned int tau)
{
  if ((J < 1) || (tau < 1))
    throw invalid_argument("J and tau must be greater than 0!");

  // Rescale the suff. stats.
  for (unsigned int k = 0; k < this->K; ++k)
  {
    this->Nk_[k] = J * this->Nk_[k] / this->J;
    this->Xk_[k] = J * this->Xk_[k] / this->J;
    this->Rk_[k] = J * this->Rk_[k] / this->J;
  }

  // Rescale the free energy contribs
  this->Fpi = J * this->Fpi / this->J;
  this->Fxz = J * this->Fxz / this->J;

  // Reset the obs counter and groups
  this->J   = J;
  this->tau = tau;

  // Reset discount and learning rates
  this->rho = 1;
  if (this->tau == 1)
    this->lambda = 0;
  else
  {
    // for  tau == 2
    this->lambda = 1 - 1 / this->tau0;
    this->rho    = 1 / (1 + this->lambda / this->rho);

    // for tau > 2, recursive
    for (unsigned int i = 2; i < this->tau; ++i)
    {
      this->lambda = 1 - (1-this->lambda) / ( 1+this->kappa * (1-this->lambda));
      this->rho    = 1 / (1 + this->lambda / this->rho);
    }
  }
}


void libcluster::IGMC::calcSS (
    const MatrixXd& X,
    const MatrixXd& qZ,
    vector<double>& Nk,
    vector<RowVectorXd>& xk,
    vector<MatrixXd>& Rk
    ) const
{
  unsigned int zK = qZ.cols();

  // Checks
  if (zK < (unsigned) this->K)
    throw invalid_argument("Invalid qZ! qZ has less columns than K!");
  if (X.cols() != this->D)
    throw invalid_argument("X must have the same dims as the suff. stats!");
  if ( (Nk.size() != zK) || (xk.size() != zK) || (Rk.size() != zK) )
    throw invalid_argument("Input vectors must be same length as qZ has cols!");

  // Initialise and precalculate stuff
  RowVectorXd Njk = qZ.colwise().sum();
  MatrixXd qZkX(X.rows(), this->D);

  // Now calculate the new suff. stats.
  for (unsigned int k = 0; k < zK; ++k)
  {
    qZkX = qZ.col(k).asDiagonal() * X;

    if (k < (unsigned) this->K)
    {
      Nk[k] = this->J*this->rho*Njk(k) + (1-this->rho)*this->Nk_[k];
      xk[k] = this->J*this->rho*qZkX.colwise().sum()+(1-this->rho)*this->Xk_[k];
      Rk[k] = this->J*this->rho*qZkX.transpose()*X + (1-this->rho)*this->Rk_[k];
    }
    else
    {
      Nk[k] = this->J*Njk(k);
      xk[k] = this->J*qZkX.colwise().sum();
      Rk[k] = this->J*qZkX.transpose()*X;
    }
  }
}


void libcluster::IGMC::calcF (double& Fpi, double& Fxz) const
{
  if (this->tau > 1)
  {
    Fpi = this->J*this->rho*Fpi + (1-this->rho)*this->Fpi;
    Fxz = this->J*this->rho*Fxz + (1-this->rho)*this->Fxz;
  }
  else
  {
    Fpi = this->J*Fpi;
    Fxz = this->J*Fxz;
  }
}


void libcluster::IGMC::delSS (unsigned int k)
{
  if (k >= this->K)
    throw invalid_argument("Invalid mixture number in k, must be [0, K-1].");

  // Clear the GMM params because they will no longer be valid
  this->w.clear();
  this->mu.clear();
  this->sigma.clear();

  // Delete sufficient stats from sufficient stat vectors
  int delidx = k;
  this->Nk_.erase(this->Nk_.begin() + delidx);
  this->Xk_.erase(this->Xk_.begin() + delidx);
  this->Rk_.erase(this->Rk_.begin() + delidx);
  --this->K;
}


void libcluster::IGMC::update (
  const MatrixXd& X,
  const MatrixXd& qZ,
  const std::vector<double>& w,
  const std::vector<Eigen::RowVectorXd>& mu,
  const std::vector<Eigen::MatrixXd>& sigma,
  double& Fpi,
  double& Fxz
  )
{
  unsigned int zK = qZ.cols();

  // Checks
  if (zK < (unsigned) this->K)
    throw invalid_argument("Invalid qZ! qZ has less columns than K!");
  if (X.cols() != this->D)
    throw invalid_argument("X must have the same dims as the suff. stats!");
  if ( (w.size() != zK) || (mu.size() != zK) || (sigma.size() != zK) )
    throw invalid_argument("Input vectors must be same length as qZ has cols!");

  // Make temp arrays
  vector<double> Nk(zK);
  vector<RowVectorXd> xk(zK);
  vector<MatrixXd> Rk(zK);

  // Get new sufficient statistic calculations
  try
    { this->calcSS(X, qZ, Nk, xk, Rk); }
  catch (invalid_argument e)
    { throw; }

  // Get new Free energy calculations
  this->calcF(Fpi, Fxz);

  // Update sufficient stats
  this->Nk_ = Nk;
  this->Xk_ = xk;
  this->Rk_ = Rk;
  this->K   = zK;

  // Update Free energy
  this->Fpi = Fpi;
  this->Fxz = Fxz;

  // Update GMM parameters
  this->w = w;
  this->mu = mu;
  this->sigma = sigma;

  // Update discount factor
  if (this->tau > 2)        // Recursive discount formula
    this->lambda = 1 - (1-this->lambda) / ( 1+this->kappa * (1-this->lambda));
  else                      // Calculate initial discount
    this->lambda = 1 - 1 / this->tau0;

  // Update learning rate
  this->rho = 1 / (1 + this->lambda / this->rho);

  ++this->tau;  // Increment observation count
}


//
// Public Functions
//

bool libcluster::learnIGMC (
  const MatrixXd& X,
  libcluster::IGMC& igmc,
  const bool verbose,
  ostream& ostrm
  )
{
  if (X.cols() != igmc.getD())
    throw invalid_argument("Dimensionality of X does not match I-GMC class!");

  // Re-create priors from the values stored in the I-GMC class
  SBprior sprior;
  GWprior gprior(igmc.getcwidthp(), igmc.getcmeanp());

  // Get an initial estimate of qZ from the I-GMC using classify if K > 1
  MatrixXd qZ, qZsplit;
  if (igmc.getK() <= 1)
    qZ = MatrixXd::Ones(X.rows(), 1);
  else
    libcluster::classifyIGMC(X, igmc, qZ);

  // Initialise Posteriors
  vector<SBposterior> spost(qZ.cols());
  vector<GWposterior> gpost(qZ.cols());

  // Initialise free energy and other loop variables
  bool changedigmc = false;
  unsigned int it;
  double Fsplt;
  Fenergy F;

  // Start VBEM learning
  if (verbose == true)
    ostrm << "Learning incremental GMC..." << endl;

  while (true)
  {
    // Run "batch" VB for this group of data
    try
      { it = vbeminc(X,qZ,igmc,sprior,gprior,spost,gpost,F,-1,verbose,ostrm); }
    catch (...)
      { throw; }

    if (it > 2)   // check if any significant VB iterations were performed
      changedigmc = true;

    // Remove any empty mixtures
    if (remempty(qZ, igmc, spost, gpost) > 0)
    {
      changedigmc = true;
      if ((verbose == true))
        ostrm << 'x' << flush;  // Notify removed some clusters
    }

    // Start looking for potential cluster splits on this group
    if (verbose == true)
      ostrm << '<' << flush;  // Notify start splitting

    try
      { Fsplt = splitinc(X,qZ,igmc,sprior,gprior,gpost,qZsplit,verbose,ostrm); }
    catch (...)
      { throw; }              // runtime_error & invalid_argument

    if (verbose == true)
      ostrm << '>' << endl;   // Notify end splitting

    // Choose either the split candidates, or finish!
    if ((Fsplt < F.tot) && (abs(F.tot-Fsplt)/F.tot > CONVERGE))
    {
      qZ = qZsplit;
      changedigmc = true;
    }
    else
      break;  // Done!
  }

  // Print finished notification if verbose
  if (verbose == true)
  {
    ostrm << "Finished!" << endl;
    if (changedigmc == true)
      ostrm << "I-GMC updated, Number of clusters = " << gpost.size() << endl;
    else
      ostrm << "I-GMC did not change significantly." << endl;
  }

  // Count the observations
  double N = 0;
  for (unsigned int k=0; k < gpost.size(); ++k)
    N += gpost[k].getNk();

  // Create GMM
  vector<RowVectorXd> mu;
  vector<MatrixXd> sigma;
  vector<double> w;

  for (unsigned int k = 0; k < gpost.size(); ++k)
  {
    mu.push_back(gpost[k].getm());
    sigma.push_back(gpost[k].getiW()/gpost[k].getnu());
    w.push_back(gpost[k].getNk()/N);
  }

  // Update I-GMC suff stats and GMM parameters
  try
    { igmc.update(X, qZ, w, mu, sigma, F.pij, F.xzj); }
  catch (invalid_argument e)
    { throw; }

  return changedigmc;
}


RowVectorXd libcluster::classifyIGMC (
  const MatrixXd& X,
  const libcluster::IGMC& igmc,
  MatrixXd& qZ,
  const bool verbose,
  ostream& ostrm
  )
{
  if (X.cols() != igmc.getD())
    throw invalid_argument("Dimensionality of X does not match I-GMC class!");
  if (igmc.getK() == 0)
    throw invalid_argument("Please learn I-GMC first by calling learnIGMC()!");
  else if (igmc.getK() == 1)
  {
    if (verbose == true)
      ostrm << "Classifying using incremental GMC: One class.";
    qZ = MatrixXd::Ones(X.rows(), 1); // Trivial clustering result
    return RowVectorXd::Ones(1);
  }

  // Re-create priors from the values stored in the I-GMC class
  SBprior sprior;
  GWprior gprior(igmc.getcwidthp(), igmc.getcmeanp());

  // Initialise Posteriors from I-GMC
  vector<SBposterior> spost(igmc.getK());
  vector<GWposterior> gpost(igmc.getK());
  try
  {
    for (int k = 0; k < igmc.getK(); ++k)
      gpost[k].update(igmc.getNk(k), igmc.getxk(k), igmc.getRk(k), gprior);
  }
  catch(...)
    { throw; }

  // Get an initial estimate of qZ from the I-GMC GMM
  qZ = libcluster::classify(X, igmc);

  // Start VBEM Classification
  if (verbose == true)
    ostrm << "Classifying using I-GMC: ";

  // Initialise free energy and other loop variables
  double F = numeric_limits<double>::max(), Fold, Fpi, Fxz,
         Ftheta = fenergyGW(gprior, gpost);
  ArrayXi order(igmc.getK());

  do
  {
    Fold = F;

    // VBM for this group
    order = vbmaximisationj(qZ, sprior, spost);

    // VBE for this group
    Fxz = vbexpectationj(X, qZ, spost, gpost, order);

    // Calculate the free energy of the model
    Fpi = fenergySBj(sprior, spost, order);
    igmc.calcF(Fpi, Fxz); // Add in discount factors for Fpi and Fxz
    F = Ftheta + Fpi + Fxz;

    if (F < 0)            // Check for bad free energy calculation
      throw runtime_error("Calculated a negative free energy!");
    if (verbose == true)  // Notify iteration
      ostrm << '-' << flush;
  }
  while (abs(Fold-F)/Fold > CONVERGE);

  // Print finished notification if verbose
  if (verbose == true)
    ostrm << " |" << endl;

  // Calculate the weights
  RowVectorXd wj(igmc.getK());

  for (int k = 0; k < igmc.getK(); ++k)
    wj(k) = exp(spost[k].getE_logZ());
  return wj;
}


//
// Private Functions
//

int vbeminc (
    const MatrixXd& X,
    MatrixXd& qZ,
    const libcluster::IGMC& igmc,
    const SBprior& sprior,
    const GWprior& gprior,
    vector<SBposterior>& spost,
    vector<GWposterior>& gpost,
    Fenergy& F,
    const int maxit,
//    const bool sparse,
    const bool verbose,
    ostream& ostrm
    )
{
  unsigned int K = qZ.cols();

  // Make sure posterior vectors are right size
  if (spost.size() != K)
    spost.resize(K, SBposterior());
  if (gpost.size() != K)
    gpost.resize(K, GWposterior());

  // Pre-allocate
  int i = 0;
  double Fold, dFpi, dFxz;
  ArrayXi order(K);

  do
  {
    Fold = F.tot;

    try
    {
      // VBM
      order = vbmaximisationj(qZ, sprior, spost);
      vbmaximisationk(X, qZ, igmc, gprior, gpost);

      // VBE for this group
      F.xzj = vbexpectationj(X, qZ, spost, gpost, order);
    }
    catch (...)
      { throw; }

    // Calculate the free energy of the model
    F.pij = fenergySBj(sprior, spost, order);
    dFpi = F.pij;
    dFxz = F.xzj;
    igmc.calcF(dFpi, dFxz); // Add in discount factors for Fpi and Fxz
    F.tot = fenergyGW(gprior, gpost) + dFpi + dFxz;

    ++i;
    if (F.tot < 0)                    // Check for bad free energy calculation
      throw runtime_error("Calculated a negative free energy!");
    if (((F.tot-Fold)/Fold > FENGYDEL) && (verbose == true)) // pos FE steps
      ostrm << '(' << (F.tot-Fold) << ')' << flush;
    if (verbose == true)              // Notify iteration
      ostrm << '-' << flush;
    if ((i >= maxit) && (maxit > 0))  // Check max iter reached
      break;
  }
  while (abs(Fold-F.tot)/Fold > CONVERGE);

  return i;
}


ArrayXi vbmaximisationj (
    const MatrixXd& qZ,
    const SBprior& sprior,
    vector<SBposterior>& spost
//    const bool sparse
    )
{
  int K  = qZ.cols(),
      Nj = qZ.rows();

  // Get cluster obs counts for this group
  RowVectorXd Njk = qZ.colwise().sum();

  // Vector for sorting the group clusters in size order
  vector<pair<int,double> > ordvec(K, pair<int,double>());  // order index

  // Record cluster size and position
  for (int k = 0; k < K; ++k)
  {
    ordvec[k].first  = k;
    ordvec[k].second = Njk(k);
  }

  // Sort the clusters in size order (descending)
  sort(ordvec.begin(), ordvec.end(), paircomp);

  // Now update the order dependent SB parameters
  ArrayXi order = ArrayXi::Zero(K);
  double Njkcumsum = 0;

  for (int k = 0; k < K; ++k)
  {
    order(k) = ordvec[k].first;         // Create row order vector
    Njkcumsum += ordvec[k].second;      // Accumulate cluster size sum
    spost[order(k)].update(ordvec[k].second, (Nj-Njkcumsum), sprior);
  }

  return order;
}


void vbmaximisationk (
    const MatrixXd& X,
    const MatrixXd& qZ,
    const libcluster::IGMC& igmc,
    const GWprior& gprior,
    vector<GWposterior>& gpost
//    const bool sparse
    )
{
  int K  = qZ.cols();

  // Calculate the new suff. stats.
  vector<double>      Nk(K);
  vector<RowVectorXd> xk(K);
  vector<MatrixXd>    Rk(K);
  igmc.calcSS(X, qZ, Nk, xk, Rk);

  // Update the GW posterior hyperparamters, and populate the order vector
  for (int k = 0; k < K; ++k)
  {
    try
      { gpost[k].update(Nk[k], xk[k], Rk[k], gprior); }
    catch(...)
      { throw; }
  }
}


double vbexpectationj (
    const MatrixXd& X,
    MatrixXd& qZ,
    vector<SBposterior>& spost,
    const vector<GWposterior>& gpost,
    const ArrayXi& order
//    const bool sparse
    )
{
  int K = qZ.cols(),
      Nj = X.rows();

  bool truncate;
  int k;
  double cumE_lognvj = 0;
  VectorXd E_logX(Nj);
  MatrixXd logqZ(Nj, K);

  // Do everything in descending size order since it is easier on memory
  for (int idx=0; idx < K; ++idx)
  {
    k = order(idx); // Get the ordered index

    // Expectations of log stick lengths (we store E_logZj in the post struct)
    truncate = (idx == (K-1)) ? true : false; // truncate the GD here?
    cumE_lognvj += spost[k].Eloglike(cumE_lognvj, truncate);

    // Expectations of log mixture likelihoods
    E_logX = gpost[k].Eloglike(X);

    // Expectations of log joint observation probs
    logqZ.col(k) = spost[k].getE_logZ() + E_logX.array();
  }

  // Log normalisation constant of log observation likelihoods
  VectorXd logZzj = logsumexp(logqZ);

  // Compute Responsabilities
  qZ = ((logqZ.colwise() - logZzj).array().exp()).matrix();

  return -logZzj.sum();
}


double fenergySBj (
    const SBprior& sprior,
    const vector<SBposterior>& spost,
    const ArrayXi& order
    )
{
  int K = spost.size();

  // If there is only one cluster, there are no SB parameters for a GD
  if (K == 1)
    return 0;

  // Free energy dependent on prior and posterior SB terms, for K-1 SB params,
  //  leaving out the last K because the SB weight is just 1.
  double Fpi = 0;

  for (int k = 0; k < (K-1); ++k)
    Fpi += spost[order(k)].fnrgpost(sprior);

  // Add in Free energy terms only dependent on priors.
  Fpi += ((K-1)*sprior.fnrgprior()); // Add in constant prior bits
  return Fpi;
}


double fenergyGW (
    const GWprior& gprior,
    const vector<GWposterior>& gpost
    )
{
  int K = gpost.size();

  // Free energy dependent on prior and posterior GW terms
  double Fn = 0;

  for (int k=0; k < K; ++k)
    Fn += gpost[k].fnrgpost(gprior);

  // Add in Free energy terms only dependent on priors.
  Fn += (K*gprior.fnrgprior()); // Add in constant prior bits
  return Fn;
}


double splitinc (
    const MatrixXd& X,
    const MatrixXd& qZ,
    const libcluster::IGMC& igmc,
    const SBprior& sprior,
    const GWprior& gprior,
    const vector<GWposterior>& gpost,
    MatrixXd& qZsplit,
//    const bool sparse
    const bool verbose,
    ostream& ostrm
    )
{
  int K = qZ.cols(),
      Nj = qZ.rows(),
      D = X.cols();

  // loop pre-allocations
  int M;
  ArrayXi mapidx;
  ArrayXb splitk;
  VectorXd eigvec(D), Njk = qZ.colwise().sum();
  MatrixXd qZk(Nj,K+1), qZr, Xk;
  Fenergy Ffree, Fbest, Ftmp;
  vector<SBposterior> spostspt(2, SBposterior()),
                      spostfree(K+1, SBposterior());
  vector<GWposterior> gpostspt(2, GWposterior()),
                      gpostfree(K+1, GWposterior());

  // Split each cluster perpendicular to P.C. and refine with VBEM. Find best
  //  split too.
  for (int k=0; k < K; ++k)
  {
    // Don't waste time with clusters that can't really be split min (2:2)
    if ((gpost[k].getNk() < 4) || (Njk(k) < 2))
      continue;

    // Make a copy of the observations with only relevant data points, p > 0.5
    partX(X, qZ.col(k), Xk, mapidx);
    M = Xk.rows();

    // Find the principle component using the power method, 'split'
    //  observations assignments, qZ, perpendicular to it.
    eigpower(gpost[k].getrefiW(), eigvec);
    splitk = (((Xk.rowwise() - gpost[k].getrefm())  // PC project and split
             * eigvec.asDiagonal()).array().rowwise().sum()) >= 0;

    // Set up VBEM for refining split
    qZr = MatrixXd::Zero(M, 2);
    qZr.col(0) = (splitk == true).cast<double>(); // Initial qZ for split
    qZr.col(1) = (splitk == false).cast<double>();

    // Refine this split using VBEM
    try
    {
      libcluster::IGMC igmcsplt(igmc, k); // Copy and use suff. stats. of cluster k
      vbeminc(Xk,qZr,igmcsplt,sprior,gprior,spostspt,gpostspt,Ftmp,SPLITITER);
    }
    catch (invalid_argument e)
      { throw invalid_argument(string("Refining split: ").append(e.what())); }
    catch (runtime_error e)
      { throw runtime_error(string("Refining split: ").append(e.what())); }

    if (checkempty<GWposterior>(gpostspt) == true)   // One cluster only
      continue;

    // Create new qZ for all data with split
    qZk = augmentqZ(k, mapidx, (qZr.col(1).array() > 0.5), qZ);

    // Calculate free energy of this split with ALL data (and refine again)
    try
      { vbeminc(X, qZk, igmc, sprior, gprior, spostfree, gpostfree, Ffree, 1); }
    catch (invalid_argument e)
      { throw invalid_argument(string("Split FE: ").append(e.what())); }
    catch (runtime_error e)
      { throw runtime_error(string("Split FE: ").append(e.what())); }

    // Just check the clusters from the split merging back to one cluster
    if ((gpostfree[k].getNk() <= 1) || (gpostfree[k+1].getNk() <= 1))
      continue;

    // Only notify here of split candidates
    if (verbose == true)
      ostrm << '=' << flush;

    // Test whether this cluster split is a keeper (free energy)
    if (Ffree.tot <= Fbest.tot)
    {
      Fbest  = Ffree;
      qZsplit = qZk;
    }
  }

  // Return best Free energy
  return Fbest.tot;
}


int remempty (
    MatrixXd& qZ,
    libcluster::IGMC& igmc,
    vector<SBposterior>& spost,
    vector<GWposterior>& gpost
  )
{
  int K = gpost.size();

  if (K <= 1)
    return 0;

  // Find empty clusters if any, return if not
  ArrayXb keepidx = ArrayXb::Ones(K);

  for (int k = 0; k < K; ++k)
    if (gpost[k].getNk() < 1)
      keepidx(k) = false;

  int keepcnt = keepidx.count();
  if (keepcnt == K)
    return 0;

  // Delete empty suff stats and reform qZ.
  MatrixXd qZnew(qZ.rows(), keepcnt);

  for (int k = 0, it = 0; k < K; ++k)
  {
    if (keepidx(k) == true)
    {
      qZnew.col(it) = qZ.col(k);
      ++it;
    }
    else
    {
      gpost.erase(gpost.begin()+it);
      spost.erase(spost.begin()+it);
      if (it <= igmc.getK())
        igmc.delSS(it);
    }
  }

  qZ = qZnew;
  return (K - keepcnt); // return the number of removed Gaussians
}
