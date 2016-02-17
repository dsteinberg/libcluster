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
    const int maxclusters,
    const bool verbose,
    const int nthreads
    );


// BGMM
boost::python::tuple wrapperBGMM (
    const boost::python::api::object& X,
    const float clusterprior,
    const int maxclusters,
    const bool verbose,
    const int nthreads
    );


// GMC
boost::python::tuple wrapperGMC (
    const boost::python::list& X,
    const float clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const int nthreads
    );


// SGMC
boost::python::tuple wrapperSGMC (
    const boost::python::list& X,
    const float clusterprior,
    const int maxclusters,
    const bool sparse,
    const bool verbose,
    const int nthreads
    );


// SCM
boost::python::tuple wrapperSCM (
    const boost::python::list& X,
    const int trunc,
    const int maxclusters,
    const float dirprior,
    const float gausprior,
    const bool verbose,
    const int nthreads
    );


// MCM
boost::python::tuple wrapperMCM (
    const boost::python::list& W,
    const boost::python::list& X,
    const int trunc,
    const int maxclusters,
    const float gausprior_t,
    const float gausprior_k,
    const bool verbose,
    const int nthreads
    );


//
//  Hack for python2/3 numpy return value weirdness
//

#if PY_MAJOR_VERSION >= 3
int*
#else
void
#endif
init_numpy()
{
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
} 


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
    "Gaussian observation model (GMC [4], SGMC/G-LDA [3, 4, 5]), and more\n"
    "highly structured models -- see the SCM and MCM functions [3, 4, 5].\n\n"
    "Author: Daniel Steinberg\n"
    "\tAustralian Centre for Field Robotics,\n"
    "\tThe University of Sydney.\n\n"
    "Date: 11/03/2013\n\n"
    "License: GPL v3 or later, See LICENSE.\n\n"
    " [1] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge,\n"
    "\tUK: pringer Science+Business Media, 2006.\n"
    " [2] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational\n"
    "\tDirichlet process mixtures, Advances in Neural Information Processing\n"
    "\tSystems, vol. 19, p. 761, 2007.\n"
    " [3] D. M. Steinberg, O. Pizarro, S. B. Williams, Synergistic Clustering\n"
    "\tof Image and Segment Descriptors for Unsupervised Scene Understanding.\n"
    "\tIn International Conference on Computer Vision (ICCV). IEEE, Sydney,\n"
    "\tNSW, 2013.\n" 
    " [4] D. M. Steinberg, O. Pizarro, S. B. Williams. Hierarchical\n"
    "\tBayesian Models for Unsupervised Scene Understanding. Journal of\n"
    "\tComputer Vision and Image Understanding (CVIU). Elsevier, 2014.\n"    
    " [5] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data,\n"
    "\tPhD Thesis, 2013.\n"
    " [6] D. M. Steinberg, A. Friedman, O. Pizarro, and S. B. Williams.\n"
    "\tA Bayesian nonparametric approach to clustering data from underwater\n"
    "\trobotic surveys. In International Symposium on Robotics Research,\n"
    "\tFlagstaff, AZ, Aug. 2011.";
  scope().attr("__doc__") = moddoc;


  // To-python converters
  init_numpy();
  to_python_converter< Eigen::ArrayXd, eigen2numpy<Eigen::ArrayXd> >();
  to_python_converter< Eigen::MatrixXd, eigen2numpy<Eigen::MatrixXd> >();
  to_python_converter< std::vector<Eigen::ArrayXd>,
                       vector2list<Eigen::ArrayXd> >();
  to_python_converter< std::vector<Eigen::MatrixXd>,
                       vector2list<Eigen::MatrixXd> >();
  to_python_converter< std::vector< std::vector<Eigen::MatrixXd> >,
                       vector2list< std::vector<Eigen::MatrixXd> > >();


  // Common documentation strings -- arguments
  const std::string comargs = "\nArguments:\n";
  const std::string Xarg =
    "\tX: array shape(N,D) the data to be clustered, N are the number of \n"
    "\t\tsamples, D the number of dimensions.\n";
  const std::string vXarg =
    "\tX: list[array shape(N_j,D),...] of len = J which is the data to be\n"
    "\t\tclustered, N_j are the number of samples of each group (or list \n"
    "\t\telement) j of data, D the number of dimensions.\n";
  const std::string vvXarg =
    "\tX: list[list[array shape(N_j,D_b),...]] where the outer list is of\n" 
    "\t\tlen = J, and each inner list is of len = I_j. This is the\n"
    "\t\t(bottom-level) data to be clustered, N_ji are the number of samples\n"
    "\t\tof each 'document/image' (ji) within each group (j) of data. D_b is\n"
    "\t\tthe number of dimensions.\n";
  const std::string truncarg = 
    "\ttrunc: the maximum number of top-level clusters to find. This is the \n"
    "\t\ttruncation level, and mostly less top-level clusters than this will\n"
    "\t\tbe returned.\n"; 
  const std::string maxclustersarg = 
    "\tmaxclusters: the maximum number of bottom level clusters to search \n"
    "\t\tfor, -1 (default) means no upper bound.";
  const std::string priorarg =
    "\tprior: the prior width of the Gaussian clusters.\n";
  const std::string priorkarg =
    "\tgausprior_k: the prior width of the bottom-level Gaussian clusters.\n";
  const std::string sparsearg =
    "\tsparse: do sparse updates? I.e. only update the clusters that have\n"
    "\t\tmore than one observation.\n";
  const std::string verbarg =
    "\tverbose: output clustering status?\n";
  const std::string threadarg =
    "\tthreads: the number of threads to use.\n";

  // Common documentation strings -- returns
  const std::string comrets = "\nReturns:\n";
  const std::string fret =
    "\tf: float, the free energy learning objective value.\n";
  const std::string qZret =
    "\tqZ: array shape(N,K), the probability of the observations belonging to\n"
    "\t\teach cluster, where K is the number of discovered clusters.\n";
  const std::string vqZret =
    "\tqZ: list[array shape(N_j,K),...] of len = J, the probability of the\n"
    "\t\tobservations in group j belonging to each cluster. Here K is the\n"
    "\t\tnumber of discovered clusters.\n";
  const std::string vvqZret =
    "\tqZ: list[list[array shape(N_j,K),...]] with the outer list of len = J,\n"
    "\t\tand each inner list of len = I_j. This is the probability of the\n"
    "\t\tbottom-level observations belonging to each cluster. Here K is the\n"
    "\t\tnumber of discovered bottom-level clusters.\n";
  const std::string vqYret =
    "\tqY: list[array shape(N_j,T),...] of len = J, the probability of the\n"
    "\t\t'documents' in group j belonging to each top-level cluster. Here T\n"
    "\t\tis the number of discovered top-level clusters.\n";
  const std::string wret =
    "\tw: array shape(K,1), the (expected) Gaussian mixture weights.\n";
  const std::string vwret =
    "\tw_j: list[array shape(K,1),...] of len = J, the (expected) Gaussian\n"
    "\t\tmixture weights of each group, j.\n";
  const std::string vwjret =
    "\tw_j: list[array shape(T,1),...] of len = J, the (expected) top-level\n"
    "\t\tcluster weights of each group, j.\n";
  const std::string vwtret =
    "\tw_t: list[array shape(K,1),...] of len = T, the (expected) Gaussian\n"
    "\t\tmixture weights of each bottom-level cluster within each of the T\n"
    "\t\ttop-level clusters.\n";
  const std::string muret =
    "\tmu: array shape(K,D), the (expected) Gaussian mixture means.\n";
  const std::string covret =
    "\tcov: list[array shape(D,D),...] of len = K, the (expected) Gaussian\n"
    "\t\t mixture covariances.\n";
  const std::string mukret =
    "\tmu_k: array shape(K,D_b), the (expected) bottom-level Gaussian mixture\n"
    "\t\tmeans.\n";
  const std::string covkret =
    "\tcov_k: list[array shape(D_b,D_b),...] of len = K, the (expected)\n"
    "\t\tbottom-level Gaussian mixture covariances.\n";


  // VDP
  const std::string vdpdoc =
    "The Variational Dirichlet Process (VDP) of [2].\n\n"
    "The VDP is similar to a regular Bayesian GMM, but places a Dirichlet\n"
    "process prior over the mixture weights. This is also used in [6].\n"
    + comargs + Xarg + priorarg + maxclustersarg + verbarg + threadarg
    + comrets + fret + qZret + wret + muret + covret;

  def ("learnVDP", wrapperVDP,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("maxclusters") = -1,
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
    "also employes the cluster splitting heuristics discussed in [2-5].\n"
    + comargs + Xarg + priorarg + maxclustersarg + verbarg + threadarg
    + comrets + fret + qZret + wret + muret + covret;

  def ("learnBGMM", wrapperBGMM,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("maxclusters") = -1,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         bgmmdoc.c_str()
      );


  // GMC
  const std::string gmcdoc =
   "The Grouped Mixtures Clustering (GMC) algorithm.\n\n"
   "This function uses the Grouped Mixtures Clustering model [5] to cluster\n"
   "multiple datasets simultaneously with cluster sharing between datasets.\n"
   "It uses a Generalised Dirichlet prior over the group mixture weights, and\n"
   "a Gaussian-Wishart prior over the cluster parameters. This algorithm is\n"
   "similar to a one-level Hierarchical Dirichlet process with Gaussian\n"
   "observations.\n"
   + comargs + vXarg + priorarg + maxclustersarg+ sparsearg + verbarg 
   + threadarg
   + comrets + fret + vqZret + vwret + muret + covret;

  def ("learnGMC", wrapperGMC,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("maxclusters") = -1,
           arg("sparse") = false,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         gmcdoc.c_str()
      );


  // SGMC
  const std::string sgmcdoc =
    "The Symmetric Grouped Mixtures Clustering (S-GMC) algorithm.\n\n"
    "This function uses the Symmetric Grouped Mixtures Clustering model [5]\n"
    "to cluster multiple datasets simultaneously with cluster sharing between\n"
    "datasets. It uses a symmetric Dirichlet prior over the group mixture\n"
    "weights, and a Gaussian-Wishart prior over the cluster parameters. This\n"
    "algorithm is similar to latent Dirichlet allocation with Gaussian\n"
    "observations.\n\n"
    "It is also referred to as Gaussian Latent Dirichlet Allocation (G-LDA)\n"
    "in [3, 4].\n"
    + comargs + vXarg + priorarg + maxclustersarg + sparsearg + verbarg 
    + threadarg
    + comrets + fret + vqZret + vwret + muret + covret;

  def ("learnSGMC", wrapperSGMC,
         (
           arg("X"),
           arg("prior") = libcluster::PRIORVAL,
           arg("maxclusters") = -1,
           arg("sparse") = false,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         sgmcdoc.c_str()
      );


  // SCM
  const std::string dpriorarg = 
    "\tdirprior: The top-level Dirichlet prior. This affects the number of\n"
    "\t\tclusters found. This may need to turned up high to have an effect.\n";

  const std::string scmdoc =
    "The Simultaneous Clustering Model (SCM).\n\n"
    "This function implements the Simultaneous Clustering Model algorithm as\n"
    "specified by [4, 5]. The SCM uses a Generalised Dirichlet prior on the\n"
    "group mixture weights, a Dirichlet prior on the top-level clusters and\n"
    "Gaussian bottom-level cluster distributions for observations (with\n"
    "Gausian-Wishart priors).\n"
    + comargs + vvXarg + truncarg + maxclustersarg + dpriorarg + priorkarg 
    + verbarg + threadarg
    + comrets + fret + vqYret + vvqZret + vwjret + vwtret + mukret + covkret;

  def ("learnSCM", wrapperSCM,
         (
           arg("X"),
           arg("trunc") = libcluster::TRUNC,
           arg("maxclusters") = -1,
           arg("dirprior") = libcluster::PRIORVAL,
           arg("gausprior") = libcluster::PRIORVAL,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         scmdoc.c_str()
       );


  // MCM
  const std::string vWarg =
    "\tW: list[array shape(I_j,D_t),...] of len = J which is the top-level\n"
    "\t\t ('document') data to be clustered, I_j are the number of documents\n"
    "\t\tin each group (or list element) j of data, D_t the number of\n"
    "\t\tdimensions.\n";
  const std::string priortarg =
    "\tgausprior_t: the prior width of the top-level Gaussian clusters.\n";
  const std::string mutret =
    "\tmu_t: array shape(T,D_t), the (expected) top-level Gaussian mixture\n"
    "\t\tmeans.\n";
  const std::string covtret =
    "\tcov_t: list[array shape(D_t,D_t),...] of len = T, the (expected)\n"
    "\t\ttop-level Gaussian mixture covariances.\n";

  const std::string mcmdoc = 
    "The Multiple-source Clustering Model (MCM).\n\n"
    "This function implements the Multiple-source Clustering Model algorithm\n"
    "as specified by [3-5]. This model jointly cluster both 'document'\n" 
    "level observations, and 'word' observations. The MCM uses a Generalised\n"
    "Dirichlet prior on the group mixture weights, Multinomial-Gaussian \n"
    "top-level (document) clusters, and Gaussian bottom-level (word) cluster\n"
    "distributions.\n"
    + comargs + vWarg + vvXarg + truncarg + maxclustersarg + priortarg
    + priorkarg + verbarg + threadarg
    + comrets + fret + vqYret + vvqZret + vwjret + vwtret + mutret + mukret 
    + covtret + covkret;

  def ("learnMCM", wrapperMCM,
         (
           arg("W"),
           arg("X"),
           arg("trunc") = libcluster::TRUNC,
           arg("maxclusters") = -1,
           arg("gausprior_t") = libcluster::PRIORVAL,
           arg("gausprior_k") = libcluster::PRIORVAL,
           arg("verbose") = false,
           arg("threads") = omp_get_max_threads()
         ),
         mcmdoc.c_str()
       );

}

#endif
