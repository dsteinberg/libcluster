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

#include <Eigen/Dense>
#include "distributions.h"
#include "libclusterpy.h"

//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace distributions;
using namespace libcluster;
using namespace boost::python;


//
// Private Functions
//


// Convert (memory share) a numpy array to an Eigen MatrixXd
MatrixXd numpy2MatrixXd (PyObject* X)
{
  if (PyArray_Check(X) == false)
    throw invalid_argument("PyObject is not an array!");
  if (PyArray_ISFLOAT(X) == false)
    throw invalid_argument("PyObject is not an array of floats/doubles!");

  return Map<MatrixXd> ((double*) PyArray_DATA(X), PyArray_DIMS(X)[0],
                   PyArray_DIMS(X)[1]);
}


// Convert (memory share) a list of numpy arrays to a vector of Eigen MatrixXd
//vMatrixXd lnumpy2vMatrixXd (list& X)
//{

//  vMatrixXd X_;
//  for (int i=0; i < len(X); ++i)
//    X_.push_back(numpy2MatrixXd(extract<PyObject*>(X[i])));

//  return X_;
//}


// Get all the means from Gaussian clusters, Kx[1xD] matrices
vMatrixXd getmean (const vector<GaussWish>& clusters)
{
  vMatrixXd means;

  for (size_t k=0; k < clusters.size(); ++k)
    means.push_back(clusters[k].getmean());

  return means;
}


// Get all of the covarances of Gaussian clusters, Kx[DxD] matrices
vMatrixXd getcov (const vector<GaussWish>& clusters)
{
  vMatrixXd covs;

  for (size_t k=0; k < clusters.size(); ++k)
    covs.push_back(clusters[k].getcov());

  return covs;
}


//
//  Public Converters
//

PyObject* ArrayXd2numpy::convert (const ArrayXd& X)
{
  npy_intp arsize[] = {X.size()};
  ArrayXd* X_ = new ArrayXd(X); // Copy to persistent array
  PyObject* Xp = PyArray_SimpleNewFromData(1, arsize, NPY_DOUBLE, X_->data());

  if (Xp == NULL)
    throw runtime_error("Cannot convert Eigen array to Numpy array!");

  return Xp;
}


PyObject* MatrixXd2numpy::convert (const MatrixXd& X)
{
  npy_intp arsize[] = {X.rows(), X.cols()};
  MatrixXd* X_ = new MatrixXd(X); // Copy to persistent array
  PyObject* Xp = PyArray_SimpleNewFromData(2, arsize, NPY_DOUBLE, X_->data());

  if (Xp == NULL)
    throw runtime_error("Cannot convert Eigen array to Numpy array!");

  return Xp;
}


PyObject* vArrayXd2numpy::convert (const vector<ArrayXd>& X)
{
  list* Xp = new list();

  for (size_t i = 0; i < X.size(); ++i)
    Xp->append(X[i]);

  return Xp->ptr();
}


PyObject* vMatrixXd2numpy::convert (const vMatrixXd& X)
{
  list* Xp = new list();

  for (size_t i = 0; i < X.size(); ++i)
    Xp->append(X[i]);

  return Xp->ptr();
}


//
//  Public Wrappers
//

// VDP
tuple wrapperVDP (
    PyObject* X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const MatrixXd X_ = numpy2MatrixXd(X);

  // Pre-allocate some stuff
  MatrixXd qZ;
  StickBreak weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnVDP(X_, qZ, weights, clusters, clusterprior, verbose,
                      nthreads);

  // Return relevant objects
  return make_tuple(f, qZ, ArrayXd(weights.Elogweight().exp()),
                    getmean(clusters), getcov(clusters));
}


// BGMM
tuple wrapperBGMM (
    PyObject* X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const MatrixXd X_ = numpy2MatrixXd(X);

  // Pre-allocate some stuff
  MatrixXd qZ;
  Dirichlet weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnBGMM(X_, qZ, weights, clusters, clusterprior, verbose,
                      nthreads);

  // Return relevant objects
  return make_tuple(f, qZ, ArrayXd(weights.Elogweight().exp()),
                    getmean(clusters), getcov(clusters));
}


// GMC
//tuple wrapperGMC (
//    const MatrixXd &X,
//    const float clusterprior,
//    const bool sparse,
//    const bool verbose,
//    const int nthreads
//    )
//{
  // Convert X
//  const vMatrixXd X_ = lnumpy2vMatrixXd(X);

//  // Pre-allocate some stuff
//  vector<MatrixXd> qZ;
//  vector<GDirichlet> weights;
//  vector<GaussWish> clusters;

//  // Do the clustering
//  double f = learnGMC(X_, qZ, weights, clusters, clusterprior, sparse, verbose,
//                      nthreads);

//  vector<ArrayXd> rwgt;
//  for (size_t k=0; k < weights.size(); ++k)
//    rwgt.push_back(ArrayXd(weights[k].Elogweight().exp()));

//  // Return relevant objects
//  return make_tuple(f, qZ, rwgt, getmean(clusters), getcov(clusters));
//}
