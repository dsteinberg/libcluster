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
vMatrixXd lnumpy2vMatrixXd (const boost::python::list& X)
{

  vMatrixXd X_;

  for (int i=0; i < len(X); ++i)
    X_.push_back(numpy2MatrixXd(X[i]));

  return X_;
}


// Convert (memory share) a list of lists of arrays to a vector of vectors of
//  matrices
vvMatrixXd llnumpy2vvMatrixXd (const boost::python::list& X)
{

  vvMatrixXd X_;

  for (int i=0; i < len(X); ++i)
  {
    vMatrixXd Xi_;

    // Compiler complains when try to use lnumpy2vmatrix instead of following
    for (int j=0; j < len(X[i]); ++j)
      Xi_.push_back(numpy2MatrixXd(X[i][j]));

    X_.push_back(Xi_);
  }

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
boost::python::tuple wrapperVDP (
    const object& X,
    const float clusterprior,
    const int maxclusters,
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
  double f = learnVDP(X_, qZ, weights, clusters, clusterprior, maxclusters,
                      verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qZ, ArrayXd(weights.Elogweight().exp()),
                                   getmean(clusters), getcov(clusters));
}


// BGMM
boost::python::tuple wrapperBGMM (
    const object& X,
    const float clusterprior,
    const int maxclusters,
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
  double f = learnBGMM(X_, qZ, weights, clusters, clusterprior, maxclusters,
                       verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qZ, ArrayXd(weights.Elogweight().exp()),
                                   getmean(clusters), getcov(clusters));
}


// GMC
boost::python::tuple wrapperGMC (
    const boost::python::list &X,
    const float clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const vMatrixXd X_ = lnumpy2vMatrixXd(X);

  // Pre-allocate some stuff
  vMatrixXd qZ;
  vector<GDirichlet> weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnGMC(X_, qZ, weights, clusters, clusterprior, maxclusters,
                      sparse, verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qZ, getweights<GDirichlet>(weights),
                                   getmean(clusters), getcov(clusters));
}


// SGMC
boost::python::tuple wrapperSGMC (
    const boost::python::list &X,
    const float clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const vMatrixXd X_ = lnumpy2vMatrixXd(X);

  // Pre-allocate some stuff
  vMatrixXd qZ;
  vector<Dirichlet> weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnSGMC(X_, qZ, weights, clusters, clusterprior, maxclusters, 
                       sparse, verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qZ, getweights<Dirichlet>(weights),
                                   getmean(clusters), getcov(clusters));
}


// SCM
boost::python::tuple wrapperSCM (
    const boost::python::list &X,
    const float dirprior,
    const float gausprior,
    const int trunc,
    const int maxclusters,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const vvMatrixXd X_ = llnumpy2vvMatrixXd(X);

  // Pre-allocate some stuff
  vMatrixXd qY;
  vvMatrixXd qZ;
  vector<GDirichlet> weights_j;
  vector<Dirichlet> weights_t;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnSCM(X_, qY, qZ, weights_j, weights_t, clusters, dirprior,
                      gausprior, trunc, maxclusters, verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qY, qZ,
                                   getweights<GDirichlet>(weights_j),
                                   getweights<Dirichlet>(weights_t),
                                   getmean(clusters), getcov(clusters));
}


// MCM
boost::python::tuple wrapperMCM (
    const boost::python::list &W,
    const boost::python::list &X,
    const float gausprior_t,
    const float gausprior_k,
    const int trunc,
    const int maxclusters,
    const bool verbose,
    const int nthreads
    )
{
  // Convert W and X
  const vMatrixXd W_ = lnumpy2vMatrixXd(W);
  const vvMatrixXd X_ = llnumpy2vvMatrixXd(X);

  // Pre-allocate some stuff
  vMatrixXd qY;
  vvMatrixXd qZ;
  vector<GDirichlet> weights_j;
  vector<Dirichlet> weights_t;
  vector<GaussWish> clusters_t;
  vector<GaussWish> clusters_k;

  // Do the clustering
  double f = learnMCM(W_, X_, qY, qZ, weights_j, weights_t, clusters_t, 
                clusters_k,  gausprior_t, gausprior_k, trunc, maxclusters,
                verbose, nthreads);

  // Return relevant objects
  return boost::python::make_tuple(f, qY, qZ,
                                   getweights<GDirichlet>(weights_j),
                                   getweights<Dirichlet>(weights_t),
                                   getmean(clusters_t), 
                                   getmean(clusters_k),
                                   getcov(clusters_t),
                                   getcov(clusters_k));
}
