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
using namespace boost::python::api;


//
// Private Functions
//


// Convert (memory share) a numpy array to an Eigen MatrixXd
MatrixXd numpy2MatrixXd (const object& X)
{
  if (PyArray_Check(X.ptr()) == false)
    throw invalid_argument("PyObject is not an array!");

  // Cast PyObject* to PyArrayObject* now we know that it's valid
  PyArrayObject* Xptr = (PyArrayObject*) X.ptr();

  if (PyArray_ISFLOAT(Xptr) == false)
    throw invalid_argument("PyObject is not an array of floats/doubles!");

  return Map<MatrixXd> ((double*) PyArray_DATA(Xptr),
                        PyArray_DIMS(Xptr)[0], PyArray_DIMS(Xptr)[1]);
}


// Convert (memory share) a list of numpy arrays to a vector of Eigen MatrixXd
vMatrixXd lnumpy2vMatrixXd (const list& X)
{

  vMatrixXd X_;
  for (int i=0; i < len(X); ++i)
    X_.push_back(numpy2MatrixXd(X[i]));

  return X_;
}


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


// Get the expected cluster weights in each of the groups
template<class W>
vector<ArrayXd> getweights (const vector<W>& weights)
{
  vector<ArrayXd> rwgt;
  for (size_t k=0; k < weights.size(); ++k)
    rwgt.push_back(ArrayXd(weights[k].Elogweight().exp()));

  return rwgt;
}


//
//  Public Wrappers
//

// VDP
tuple wrapperVDP (
    const object& X,
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
    const object& X,
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
tuple wrapperGMC (
    const list &X,
    const float clusterprior,
    const bool sparse,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const vMatrixXd X_ = lnumpy2vMatrixXd(X);

  // Pre-allocate some stuff
  vector<MatrixXd> qZ;
  vector<GDirichlet> weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnGMC(X_, qZ, weights, clusters, clusterprior, sparse, verbose,
                      nthreads);

  // Return relevant objects
  return make_tuple(f, qZ, getweights<GDirichlet>(weights), getmean(clusters),
                    getcov(clusters));
}


// SGMC
tuple wrapperSGMC (
    const list &X,
    const float clusterprior,
    const bool sparse,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const vMatrixXd X_ = lnumpy2vMatrixXd(X);

  // Pre-allocate some stuff
  vector<MatrixXd> qZ;
  vector<Dirichlet> weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnSGMC(X_, qZ, weights, clusters, clusterprior, sparse, verbose,
                      nthreads);

  // Return relevant objects
  return make_tuple(f, qZ, getweights<Dirichlet>(weights), getmean(clusters),
                    getcov(clusters));
}
