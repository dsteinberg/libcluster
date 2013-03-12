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

#include <omp.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "libcluster.h"

//
//  To-python type converters
//

// ArrayXd to numpy array ([...])
struct ArrayXd2numpy
{
  static PyObject* convert (const Eigen::ArrayXd& X);
};


// MatrixXd to numpy array ([[..]])
struct MatrixXd2numpy
{
  static PyObject* convert (const Eigen::MatrixXd& X);
};


// vector<ArrayXd> to list of numpy arrays [([...]), ([...]), ...]
struct vArrayXd2numpy
{
  static PyObject* convert (const std::vector<Eigen::ArrayXd>& X);
};


// vector<MatrixXd> to list of numpy arrays [([[...]]), ([[...]]), ...]
struct vMatrixXd2numpy
{
  static PyObject* convert (const libcluster::vMatrixXd& X);
};


//
//  Wrappers
//

// VDP
boost::python::tuple wrapperVDP (
    PyObject* X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    );


// BGMM
boost::python::tuple wrapperBGMM (
    PyObject* X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    );


// GMC
//boost::python::tuple wrapperGMC (
//    const Eigen::MatrixXd& X,
//    const float clusterprior,
//    const bool sparse,
//    const bool verbose,
//    const int nthreads
//    );


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
  to_python_converter<Eigen::ArrayXd, ArrayXd2numpy>();
  to_python_converter<Eigen::MatrixXd, MatrixXd2numpy>();
  to_python_converter<libcluster::vMatrixXd, vMatrixXd2numpy>();


  // Common documentation
  const std::string comargs =
    "Arguments:\n"
    "\tX: array shape(N,D) the data to be clustered, N are the number of \n"
    "\t\tsamples, D the number of dimensions.\n"
    "\tprior: float (1e-5), the prior width of the Gaussian clusters.\n"
    "\tverbose: bool (False), output clustering status?\n"
    "\tthreads: int (number of cores), the number of threads to use.";


  // VDP
  const std::string vdpdoc =
    "The Variational Dirichlet Process (VDP) of [2].\n\n"
    "The VDP is similar to a regular Bayesian GMM, but places a Dirichlet\n"
    "process prior over the mixture weights.\n\n" + comargs;

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
    "\n" + comargs;

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
//  const std::string gmcdoc = "";

//  def ("learnGMC", wrapperGMC,
//         (
//           arg("X"),
//           arg("prior") = libcluster::PRIORVAL,
//           arg("sparse") = false,
//           arg("verbose") = false,
//           arg("threads") = omp_get_max_threads()
//         ),
//         gmcdoc.c_str()
//      );

}

#endif
