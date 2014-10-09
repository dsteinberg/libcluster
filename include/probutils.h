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

#ifndef PROBUTILS_H
#define PROBUTILS_H

#include <stdexcept>
#include <Eigen/Dense>
#include <vector>


//
// Namespaces
//

/*! \brief Namespace for various linear algebra tools useful for dealing with
 *         Gaussians and log-probability expressions.
 *
 * \author Daniel Steinberg
 *         Australian Centre for Field Robotics
 *         The University of Sydney
 *
 * \date   15/02/2011
 */
namespace probutils
{


//
// Useful Functions
//

/*! \brief Calculate the column means of a matrix.
 *
 *  \param X an NxD matrix.
 *  \returns a 1xD row vector of the means of each column of X.
 */
Eigen::RowVectorXd mean (const Eigen::MatrixXd& X);


/*! \brief Calculate the column means of a vector of matrices (one mean for
 *         all data in the matrices).
 *
 *  \param X a vector of N_jxD matrices for j = 1:J.
 *  \returns a 1xD row vector of the means of each column of X.
 *  \throws std::invalid_argument if X has inconsistent D between elements.
 */
Eigen::RowVectorXd mean (const std::vector<Eigen::MatrixXd>& X);


/*! \brief Calculate the column standard deviations of a matrix, uses N - 1.
 *
 *  \param X an NxD matrix.
 *  \returns a 1xD row vector of the standard deviations of each column of X.
 */
Eigen::RowVectorXd stdev (const Eigen::MatrixXd& X);


/*! \brief Calculate the covariance of a matrix.
 *
 *    If X is an NxD matrix, then this calculates:
 *
 *    \f[ Cov(X) = \frac{1} {N-1} (X-E[X])^T (X-E[X]) \f]
 *
 *  \param X is a NxD matrix to calculate the covariance of.
 *  \returns a DxD covariance matrix.
 *  \throws std::invalid_argument if X is 1xD or less (has one or less
 *          observations).
 */
Eigen::MatrixXd cov (const Eigen::MatrixXd& X);


/*! \brief Calculate the covariance of a vector of matrices (one mean for
 *         all data in the matrices).
 *
 *    This calculates:
 *
 *    \f[ Cov(X) = \frac{1} {\sum_j N_j-1}  \sum_j (X_j-E[X])^T (X_j-E[X]) \f]
 *
 *  \param X is a a vector of N_jxD matrices for j = 1:J.
 *  \returns a DxD covariance matrix.
 *  \throws std::invalid_argument if any X_j has one or less observations.
 *  \throws std::invalid_argument if X has inconsistent D between elements.
 */
Eigen::MatrixXd cov (const std::vector<Eigen::MatrixXd>& X);


/*! \brief Calculate the Mahalanobis distance, (x-mu)' * A^-1 * (x-mu), N
 *         times.
 *
 *  \param X an NxD matrix of samples/obseravtions.
 *  \param mu a 1XD vector of means.
 *  \param A a DxD marix of weights, A must be invertable.
 *  \returns an Nx1 matrix of distances evaluated for each row of X.
 *  \throws std::invalid_argument If X, mu and A do not have compatible
 *          dimensionality, or if A is not PSD.
 */
Eigen::VectorXd mahaldist (
    const Eigen::MatrixXd& X,
    const Eigen::RowVectorXd& mu,
    const Eigen::MatrixXd& A
    );


/*! \brief Perform a log(sum(exp(X))) in a numerically stable fashion.
 *
 *  \param X is a NxK matrix. We wish to sum along the rows (sum out K).
 *  \returns an Nx1 vector where the log(sum(exp(X))) operation has been
 *           performed along the rows.
 */
Eigen::VectorXd logsumexp (const Eigen::MatrixXd& X);


/*! \brief The eigen power method. Return the principal eigenvalue and
 *         eigenvector.
 *
 *  \param A is the square DxD matrix to decompose.
 *  \param eigvec is the Dx1 principal eigenvector (mutable)
 *  \returns the principal eigenvalue.
 *  \throws std::invalid_argument if the matrix A is not square
 *
 */
double eigpower (const Eigen::MatrixXd& A, Eigen::VectorXd& eigvec);


/*! \brief Get the log of the determinant of a PSD matrix.
 *
 *  \param A a DxD positive semi-definite matrix.
 *  \returns log(det(A))
 *  \throws std::invalid_argument if the matrix A is not square or if it is
 *          not positive semidefinite.
 */
double logdet (const Eigen::MatrixXd& A);


/*! \brief Calculate digamma(X) for each element of X.
 *
 *  \param X an NxM matrix
 *  \returns an NxM matrix for which digamma(X) has been calculated for each
 *           element
 */
Eigen::MatrixXd mxdigamma (const Eigen::MatrixXd& X);


/*! \brief Calculate log(gamma(X)) for each element of X.
 *
 *  \param X an NxM matrix
 *  \returns an NxM matrix for which log(gamma(X)) has been calculated for
 *           each element
 */
Eigen::MatrixXd mxlgamma (const Eigen::MatrixXd& X);

}

#endif // PROBUTILS_H
