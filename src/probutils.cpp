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
  const int J = X.size(),
            D = X[0].cols();
  int N = 0;
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
  const int J = X.size(),
            D = X[0].cols();
  int N = 0;
  const RowVectorXd mean = probutils::mean(X);
  MatrixXd cov = MatrixXd::Zero(D, D),
           X_mu;

  for (int j = 0; j < J; ++j)
  {
    if (X[j].rows() <= 1)
      throw invalid_argument("Insufficient no. of observations.");
    X_mu = X[j].rowwise() - mean;
    N   += X[j].rows();
    cov.noalias() += (X_mu.transpose() * X_mu); // (X_j-mu)'*(X_j-mu)
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

  // Decompose A
  LDLT<MatrixXd> Aldl(A);

  // Check if A is PD
  if ((Aldl.vectorD().array() <= 0).any() == true)
    throw invalid_argument("Matrix A is not positive definite");

  // Do the Mahalanobis distance for each sample (N times)
  MatrixXd X_mu = (X.rowwise() - mu).transpose();
  return ((X_mu.array() * (Aldl.solve(X_mu)).array())
          .colwise().sum()).transpose();
}


VectorXd probutils::logsumexp (const MatrixXd& X)
{
  const VectorXd mx = X.rowwise().maxCoeff(); // Get max of each row

  // Perform the sum(exp(x - mx)) part
  ArrayXd se = ((X.colwise() - mx).array().exp()).rowwise().sum();

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
    eigvec.setOnes(1);
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
    v.noalias() = A * oeigvec;
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

  VectorXd d = A.ldlt().vectorD();  // Get the diagonal from a Cholesky decomp.

  // Check if A is PD
  if ((d.array() <= 0).any() == true)
    throw domain_error("Matrix A is not positive definite.");

  return (d.array().log()).sum();   // ln(det(A)) = sum(log(d))
}


MatrixXd probutils::mxdigamma (const MatrixXd& X)
{
  const int I = X.rows(),
            J = X.cols();
  MatrixXd result(I, J);

  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      result(i,j) = boost::math::digamma(X(i, j));

  return result;
}


MatrixXd probutils::mxlgamma (const MatrixXd& X)
{
  const int I = X.rows(),
            J = X.cols();
  MatrixXd result(I, J);

  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      result(i, j) = boost::math::lgamma(X(i, j));

  return result;
}
