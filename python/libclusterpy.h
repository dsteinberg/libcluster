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

#ifndef LIBCLUSTERPY_H
#define LIBCLUSTERPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  // Test deprication for v1.7

#include <omp.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "libcluster.h"


//
//  To-python type converters
//

// Eigen::MatrixXd/ArrayXd (double) to numpy array ([[...]])
template<typename M>
struct eigen2numpy
{
  static PyObject* convert (const M& X)
  {
    npy_intp arsize[] = {X.rows(), X.cols()};
    M* X_ = new M(X); // Copy to persistent array
    PyObject* Xp = PyArray_SimpleNewFromData(2, arsize, NPY_DOUBLE, X_->data());

    if (Xp == NULL)
      throw std::runtime_error("Cannot convert Eigen matrix to Numpy array!");

    return Xp;
  }
};


// std::vector<Something> to python list [...].
template<typename M>
struct vector2list
{
  static PyObject* convert (const std::vector<M>& X)
  {
    boost::python::list* Xp = new boost::python::list();

    for (size_t i = 0; i < X.size(); ++i)
      Xp->append(X[i]);

    return Xp->ptr();
  }
};


//
//  Wrappers
//

// VDP
boost::python::tuple wrapperVDP (
    const boost::python::api::object& X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    );


// BGMM
boost::python::tuple wrapperBGMM (
    const boost::python::api::object& X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    );


// GMC
boost::python::tuple wrapperGMC (
    const boost::python::list& X,
    const float clusterprior,
    const bool sparse,
    const bool verbose,
    const int nthreads
    );


// SGMC
boost::python::tuple wrapperSGMC (
    const boost::python::list& X,
    const float clusterprior,
    const bool sparse,
    const bool verbose,
    const int nthreads
    );


//
//  Module definition
//

BOOST_PYTHON_MODULE (libclusterpy)
{
  using namespace boost::python;

  // This will enable user-defined docstrings and python signatures,
  // while disabling the C++ signatures
  docstring_options local_docstring_options(true, true, false);


  // set the docstring of the current module scope
  const std::string moddoc =
    "A collection of structured Bayesian clustering algorithms.\n\n"
    "This library contains implementations of a number of variational\n"
    "Bayesian clustering algorithms such as the Bayesian Gaussian Mixture\n"
    "model of [1], and the Variational Dirichlet process of [2]. Also \n"
    "implemented is a latent Dirichlet allocation-like model with a \n"
    "Gaussian observation model, and even more highly structured models --\n"
    "see the SCM and MCM functions [3].\n\n"
    "Author: Daniel Steinberg\n"
    "\tAustralian Centre for Field Robotics, The University of Sydney.\n"
    "Date: 11/03/2013\n"
    "License: GPL v3 or later, See LICENSE.\n\n"
    "[1] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge,\n"
    "\tUK: pringer Science+Business Media, 2006.\n"
    "[2] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational\n"
    "\tDirichlet process mixtures, Advances in Neural Information Processing\n"
    "\tSystems, vol. 19, p. 761, 2007.\n"
    "[3] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data,\n"
    "\tPhD Thesis, 2013.";
  scope().attr("__doc__") = moddoc;


  // To-python converters
  import_array();
  to_python_converter< Eigen::ArrayXd, eigen2numpy<Eigen::ArrayXd> >();
  to_python_converter< Eigen::MatrixXd, eigen2numpy<Eigen::MatrixXd> >();
  to_python_converter< std::vector<Eigen::ArrayXd>,
                       vector2list<Eigen::ArrayXd> >();
  to_python_converter< std::vector<Eigen::MatrixXd>,
                       vector2list<Eigen::MatrixXd> >();


  // Common documentation strings -- arguments
  const std::string comargs = "\nArguments:\n";
  const std::string Xarg =
    "\tX: array shape(N,D) the data to be clustered, N are the number of \n"
    "\t\tsamples, D the number of dimensions.\n";
  const std::string vXarg =
    "\tX: list[array shape(N_j,D),...] of len = J which is the data to be\n"
    "\t\tclustered, N_j are the number of samples of each group (or list \n"
    "\t\telement) j of data, D the number of dimensions.\n";
  const std::string priorarg =
    "\tprior: float (1.0), the prior width of the Gaussian clusters.\n";
  const std::string sparsearg =
    "\tsparse: bool (False), do sparse updates? I.e. only update the clusters\n"
    "\t\tthat have more than one observation.\n";
  const std::string verbarg =
    "\tverbose: bool (False), output clustering status?\n";
  const std::string threadarg =
    "\tthreads: int (number of cores), the number of threads to use.\n";

  // Common documentation strings -- returns
  const std::string comrets = "\nReturns:\n";
  const std::string fret =
    "\tf: float, the free energy learning objective value.\n";
  const std::string qZret =
    "\tqZ: array shape(N,K), the probability of the observations belonging to\n"
    "\t\teach cluster, where K is the number of discovered clusters.\n";
  const std::string vqZret =
    "\tqZ: list[array shape(N_j,K),...] of len = J, the probability of the\n"
    "\t\tobservations in group, j, belonging to each cluster. Here K is the\n"
    "\t\tnumber of discovered clusters.\n";
  const std::string wret =
    "\t w: array shape(K,1), the (expected) Gaussian mixture weights.\n";
  const std::string vwret =
    "\tw: list[array shape(K,1),...] of len = J, the (expected) Gaussian\n"
    "\t\tmixture weights of each group, j.\n";
  const std::string muret =
    "\tmu: array shape(K,D), the (expected) Gaussian mixture means.\n";
  const std::string covret =
    "\tcov: list[array shape(D,D),...] of len = K, the (expected) Gaussian\n"
    "\t\t mixture covariances.\n";


  // VDP
  const std::string vdpdoc =
    "The Variational Dirichlet Process (VDP) of [2].\n\n"
    "The VDP is similar to a regular Bayesian GMM, but places a Dirichlet\n"
    "process prior over the mixture weights.\n"
    + comargs + Xarg + priorarg + verbarg + threadarg
    + comrets + fret + qZret + wret + muret + covret;

  def ("learnVDP", wrapperVDP,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         vdpdoc.c_str()
      );


  // BGMM
  const std::string bgmmdoc =
    "The Bayseian Gaussian mixture model (BGMM) described in [1].\n\n"
    "This BGMM is similar to a GMM learned with EM, but it places a\n"
    "Dirichlet prior over the mixture weights, and Gaussian-Wishart priors\n"
    "over the Gaussian clusters. This implementation is similar to [1] but\n"
    "also employes the cluster splitting heuristics discussed in [2] and [3].\n"
    + comargs + Xarg + priorarg + verbarg + threadarg
    + comrets + fret + qZret + wret + muret + covret;

  def ("learnBGMM", wrapperBGMM,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         bgmmdoc.c_str()
      );


  // GMC
  const std::string gmcdoc =
   "The Grouped Mixtures Clustering (GMC) algorithm.\n\n"
   "This function uses the Grouped Mixtures Clustering model [3] to cluster\n"
   "multiple datasets simultaneously with cluster sharing between datasets.\n"
   "It uses a Generalised Dirichlet prior over the group mixture weights, and\n"
   "a Gaussian-Wishart prior over the cluster parameters. This algorithm is\n"
   "similar to a one-level Hierarchical Dirichlet process with Gaussian\n"
   "observations.\n"
   + comargs + vXarg + priorarg + sparsearg + verbarg + threadarg
   + comrets + fret + vqZret + vwret + muret + covret;

  def ("learnGMC", wrapperGMC,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("sparse") = false,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         gmcdoc.c_str()
      );


  // SGMC
  const std::string sgmcdoc = ""
    "The Symmetric Grouped Mixtures Clustering (S-GMC) algorithm.\n\n"
    "This function uses the Symmetric Grouped Mixtures Clustering model [3]\n"
    "to cluster multiple datasets simultaneously with cluster sharing between\n"
    "datasets. It uses a symmetric Dirichlet prior over the group mixture\n"
    "weights, and a Gaussian-Wishart prior over the cluster parameters. This\n"
    "algorithm is similar to latent Dirichlet allocation with Gaussian\n"
    "observations.\n"
    + comargs + vXarg + priorarg + sparsearg + verbarg + threadarg
    + comrets + fret + vqZret + vwret + muret + covret;

  def ("learnSGMC", wrapperSGMC,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("sparse") = false,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         sgmcdoc.c_str()
      );

}

#endif
