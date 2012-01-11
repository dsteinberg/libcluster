// TODO
// - make mxdigamma and mxlgamma work on all eigen matrix and array types!

#include "probutils.h"
#include <boost/math/special_functions.hpp>


//
// Namespaces
//


using namespace std;
using namespace Eigen;


//
// Local Constants
//


const double EIGCONTHRESH = 1.0e-8f;
const int    MAXITER      = 100;


//
// Public Functions
//


RowVectorXd probutils::mean (const MatrixXd& X)
{
  return X.colwise().sum()/X.rows();
}


RowVectorXd probutils::mean (const vector<MatrixXd>& X)
{
  int J = X.size(), D = X[0].cols(), N = 0;
  RowVectorXd mean = RowVectorXd::Zero(D);

  for (int j = 0; j < J; ++j)
  {
    if (X[j].cols() != D)
      throw invalid_argument("X dimensions are inconsistent between groups!");

    mean += X[j].colwise().sum();
    N    += X[j].rows();
  }
  return mean / N;
}


RowVectorXd probutils::stdev (const MatrixXd& X)
{
  RowVectorXd meanX = mean(X);
  return ((X.rowwise() - meanX).array().square().colwise().sum()
          / (X.rows()-1)).sqrt();
}


MatrixXd probutils::cov (const MatrixXd& X)
{
  if (X.rows() <= 1)
    throw invalid_argument("Insufficient no. of observations.");

  MatrixXd X_mu = X.rowwise() - probutils::mean(X); // X - mu
  return (X_mu.transpose()*X_mu)/(X.rows()-1);      // (X-mu)'*(X-mu)/(N-1)
}


MatrixXd probutils::cov (const vector<MatrixXd>& X)
{
  int J = X.size(), D = X[0].cols(), N = 0;
  RowVectorXd mean = probutils::mean(X);
  MatrixXd cov = MatrixXd::Zero(D, D), X_mu;

  for (int j = 0; j < J; ++j)
  {
    if (X[j].rows() <= 1)
      throw invalid_argument("Insufficient no. of observations.");
    X_mu = X[j].rowwise() - mean;
    N   += X[j].rows();
    cov.noalias() += (X_mu.transpose()*X_mu); // (X_j-mu)'*(X_j-mu)
  }

  return cov / (N-1);
}


VectorXd probutils::mahaldist (
    const MatrixXd& X,
    const RowVectorXd& mu,
    const MatrixXd& A
    )
{
  // Check for same number of dimensions, D
  if((X.cols() != mu.cols()) || (X.cols() != A.cols()))
    throw invalid_argument("Arguments do not have the same dimensionality");
  // Check if A is square
  if (A.rows() != A.cols())
    throw invalid_argument("Matrix A must be square!");
  // Check if A is PSD
  if (A.ldlt().isPositive() == false)
    throw invalid_argument("Matrix A is not positive semidefinite");

  // Do the Mahalanobis distance for each sample (N times)
  MatrixXd X_mu = (X.rowwise() - mu).transpose();
  return (( X_mu.array() * (A.ldlt().solve(X_mu)).array() )
          .colwise().sum() ).transpose();
}


VectorXd probutils::logsumexp (const MatrixXd& X)
{
  VectorXd mx = X.rowwise().maxCoeff(); // Get max of each row
  ArrayXd se(X.rows());

  // Perform the sum(exp(x - mx)) part
  se = ((X.colwise() - mx).array().exp()).rowwise().sum();

  // return total log(sum(exp(x))) - hoping for return value optimisation
  return (se.log()).matrix() + mx;
}


double probutils::eigpower (const MatrixXd& A, VectorXd& eigvec)
{
  // Check if A is square
  if (A.rows() != A.cols())
    throw invalid_argument("Matrix A must be square!");

  // Check if A is a scalar
  if (A.rows() == 1)
  {
    eigvec = VectorXd::Ones(1);
    return A(0,0);
  }

  // Initialise working vectors
  VectorXd v = VectorXd::LinSpaced(A.rows(), -1, 1);
  VectorXd oeigvec(A.rows());

  // Initialise eigenvalue and eigenvectors etc
  double eigval = v.norm();
  double vdist = numeric_limits<double>::infinity();
  eigvec = v/eigval;

  // Loop until eigenvector converges or we reach max iterations
  for (int i=0; (vdist>EIGCONTHRESH) && (i<MAXITER); ++i)
  {
    oeigvec = eigvec;
    v.noalias() = A*oeigvec;
    eigval = v.norm();
    eigvec = v/eigval;
    vdist = (eigvec - oeigvec).norm();
  }

  return eigval;
}


double probutils::logdet (const MatrixXd& A)
{
  // Check if A is square
  if (A.rows() != A.cols())
    throw invalid_argument("Matrix A must be square!");
  // Check if A is PSD
  if (A.ldlt().isPositive() == false)
    throw invalid_argument("Matrix A is not positive semidefinite");

  VectorXd d = A.ldlt().vectorD();  // Get the diagonal from an LDL decomp
  return (d.array().log()).sum();   // ln(det(A)) = sum(log(d))
}


MatrixXd probutils::mxdigamma (const MatrixXd& X)
{
  int I = X.rows();
  int J = X.cols();
  MatrixXd result(I,J);

  for (int i=0; i < I; ++i)
    for (int j=0; j < J; ++j)
      result(i,j) = boost::math::digamma(X(i,j));

  return result;
}


MatrixXd probutils::mxlgamma (const MatrixXd& X)
{
  int I = X.rows();
  int J = X.cols();
  MatrixXd result(I,J);

  for (int i=0; i < I; ++i)
    for (int j=0; j < J; ++j)
      result(i,j) = boost::math::lgamma(X(i,j));

  return result;
}


double probutils::cseperation (
    double eigvalk,
    double eigvall,
    const RowVectorXd& muk,
    const RowVectorXd& mul
    )
{
  if (muk.cols() != mul.cols())
    throw invalid_argument("muk and mul must have the same dimensionality!");

  return (muk - mul).array().square().sum()
      / ( muk.cols() * max<double>(eigvalk, eigvall));
}
