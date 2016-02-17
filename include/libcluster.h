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

#ifndef LIBCLUSTER_H
#define LIBCLUSTER_H

#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <omp.h>
#include "distributions.h"


//
// Namespace Definitions
//

/*! \brief Namespace that contains implementations of Bayesian mixture model
 *         based algorithms for clustering.
 *
 *  This namespace provides various Bayesian mixture models that can be used
 *  for clustering data. The algorithms that have been implemented are:
 *
 *    - Variational Dirichlet Process (VDP) for Gaussian observations [1, 6], 
 *      see learnVDP().
 *    - The Bayesian Gaussian Mixture model [7] ch 11, see learnBGMM().
 *    - The Bayesian Gaussian Mixture model with diagonal covariance Gaussians,
 *      see learnDGMM().
 *    - Bayesian Exponential Mixture model with a Gamma prior, see learnBEMM().
 *    - Groups of Mixtures Clustering (GMC) model for Gaussian observations
 *      [4], see learnGMC().
 *    - Symmetric Groups of Mixtures Clustering (S-GMC) model for Gaussian
 *      observations [5], see learnSGMC(). This is referred to as Gaussian
 *      Latent Dirichlet Allocation (G-LDA) in [3, 4].
 *    - Groups of Mixtures Clustering model for diagonal covariance Gaussian
 *      observations, see learnDGMC().
 *    - Groups of Mixtures Clustering model for Exponential observations, see
 *      learnEGMC().
 *    - Simultaneous Clustering Model (SCM) for Multinomial Documents, and
 *      Gaussian Observations, see learnSCM() and [4, 5].
 *    - Multiple-source Clustering Model (MCM) for clustering two observations,
 *      one of an image/document, and multiple of segments/words
 *      simultaneously, see learnMCM() and [3, 4, 5].
 *    - A myriad of other algorithms are possible, but have not been enumerated
 *      in the interfaces here.
 *
 *  All of these algorithms infer the number of clusters present in the data.
 *
 * [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
 *     Dirichlet process mixtures. Advances in Neural Information Processing
 *     Systems, vol. 19, p. 761, 2007.
 *
 * [2] Y. Teh, K. Kurihara, and M. Welling. Collapsed variational inference
 *     for HDP. Advances in Neural Information Processing Systems,
 *     20:1481–1488, 2008.
 *
 * [3] D. M. Steinberg, O. Pizarro, S. B. Williams. Synergistic Clustering of
 *     Image and Segment Descriptors for Unsupervised Scene Understanding.
 *     In International Conference on Computer Vision (ICCV). IEEE, Sydney,
 *     NSW, 2013.
 *
 * [4] D. M. Steinberg, O. Pizarro, S. B. Williams. Hierarchical Bayesian
 *     Models for Unsupervised Scene Understanding. Journal of Computer Vision
 *     and Image Understanding (CVIU). Elsevier, 2014.
 *
 * [5] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
 *     Thesis, 2013.
 *
 * [6] D. M. Steinberg, A. Friedman, O. Pizarro, and S. B. Williams. A Bayesian 
 *     nonparametric approach to clustering data from underwater robotic
 *     surveys. In International Symposium on Robotics Research, Flagstaff, AZ, 
 *     Aug. 2011.
 *
 * [7] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
 *     Springer Science+Business Media, 2006.
 *
 * \note The greedy cluster splitting heuristic is different from that 
 *       presented in [1] in that it is much faster, but may not choose the 
 *       "best" cluster to split first.
 *
 * \note The code is generic enough to allow new clustering algorithms to be
 *       implemented quickly, since all of the algorithms use templated
 *       distribution types.
 *
 * \author Daniel Steinberg
 *         Australian Centre for Field Robotics
 *         The University of Sydney
 *
 * \date   20/10/2013
 *
 * \todo Find a better way to parallelise the vanilla clustering algorithms.
 * \todo Make this library more generic so discrete distributions can be used?
 * \todo Should probably get rid of all the vector copies in splitting
 *       functions and interface functions.
 */
namespace libcluster
{


//
// Namespace constants
//

const double       PRIORVAL   = 1.0;     //!< Default prior hyperparameter value
const unsigned int TRUNC      = 100;     //!< Truncation level for classes
const unsigned int SPLITITER  = 15;      //!< Max number of iter. for split VBEM
const double       CONVERGE   = 1e-5f;       //!< Convergence threshold
const double       FENGYDEL   = CONVERGE/10; //!< Allowance for +ve F.E. steps
const double       ZEROCUTOFF = 0.1f;    //!< Obs. count cut off sparse updates


//
// Convenience Typedefs
//

//! Vector of double matricies
typedef std::vector<Eigen::MatrixXd>                  vMatrixXd;

//! Vector of vectors of double matricies
typedef std::vector< std::vector<Eigen::MatrixXd> >   vvMatrixXd;


//
// Mixture Models for Clustering (cluster.cpp)
//

/*! \brief The learning algorithm for the Variational Dirichlet Process for
 *         Gaussian clusters.
 *
 * This function implements the VDP clustering algorithm as specified by [1],
 * however a different 'nesting' strategy is used. The nesting strategy sets all
 * q(z_n > K) = 0, rather than setting the parameter distributions equal to
 * their priors over this truncation bound, K. This is the same nesting strategy
 * as used in [2]. This is also used in [3-5].
 *
 *  \param X the observation matrix, NxD where N is the number of observations,
 *         and D is the number of dimensions.
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param weights is the distributions over the mixture weights of the model.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The group cluster algorithms take fuller advantage of this. The
 *         default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnVDP (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    distributions::StickBreak& weights,
    std::vector<distributions::GaussWish>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for a Bayesian Gaussian Mixture model.
 *
 * This function implements the Bayesian GMM clustering algorithm as specified
 * by [1]. In practice I have found this performs almost identically to the VDP,
 * especially for large data cases.
 *
 *  \param X the observation matrix, NxD where N is the number of observations,
 *         and D is the number of dimensions.
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param weights is the distributions over the mixture weights of the model.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The group cluster algorithms take fuller advantage of this. The
 *         default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnBGMM (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    distributions::Dirichlet& weights,
    std::vector<distributions::GaussWish>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for a Bayesian Gaussian Mixture model with
 *         diagonal covariance matrices.
 *
 * This function implements the Bayesian GMM clustering algorithm as specified
 * by [1] but with diagonal covariance matrices, i.e. this is a Naive-Bayes
 * assumption.
 *
 *  \param X the observation matrix, NxD where N is the number of observations,
 *         and D is the number of dimensions.
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param weights is the distributions over the mixture weights of the model.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The group cluster algorithms take fuller advantage of this. The
 *         default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          negative diagonal covariance matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnDGMM (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    distributions::Dirichlet& weights,
    std::vector<distributions::NormGamma>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for a Bayesian Exponential Mixture model.
 *
 * This function implements a Bayesian Exponential mixture model clustering
 * algorithm. The Exponential mixture model uses a Dirichlet prior on the
 * mixture weights, but an Exponential cluster distribution (with a Gamma
 * prior). Each dimension of the data is assumed independent i.e. this is a
 * Naive-Bayes assumption.
 *
 *  \param X the observation matrix, NxD where N is the number of observations,
 *         and D is the number of dimensions. X MUST be in the range [0, inf).
 *  \param qZ is an NxK matrix of the variational posterior approximation to
 *         p(Z|X). This will always be overwritten to start with one
 *         cluster.
 *  \param weights is the distributions over the mixture weights of the model.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The group cluster algorithms take fuller advantage of this. The
 *         default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls.
 *  \throws std::runtime_error if there are runtime issues with the VDP
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnBEMM (
    const Eigen::MatrixXd& X,
    Eigen::MatrixXd& qZ,
    distributions::Dirichlet& weights,
    std::vector<distributions::ExpGamma>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for the Groups of Mixtures Clustering model.
 *
 * This function implements the Groups of Mixtues Clustering model algorithm
 * as specified by [4], with the additional "sparse" option. The GMC uses a
 * Generalised Dirichlet prior on the group mixture weights and Gaussian cluster
 * distributions (With Gausian-Wishart priors). This algorithm is similar to a
 * one-level Hierarchical Dirichlet process with Gaussian observations.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions.
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param weights is a vector of distributions over the mixture weights of the
 *         model, for each group of data, J.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    std::vector<distributions::GDirichlet>& weights,
    std::vector<distributions::GaussWish>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool sparse = false,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for the Symmetric Groups of Mixtures
 *         Clustering model. The is referred to as *Gaussian Latent Dirichlet
 *         Allocation* (G-LDA) in [3, 4].
 *
 * This function implements the Symmetric Groups of Mixtures Clustering model
 * as specified by [4, 5], with the additional "sparse" option. The Symmetric 
 * GMC uses a symmetric Dirichlet prior on the group mixture weights and
 * Gaussian cluster distributions (With Gausian-Wishart priors). This algorithm
 * is similar to latent Dirichlet allocation with Gaussian observations.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions.
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param weights is a vector of distributions over the mixture weights of the
 *         model, for each group of data, J.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnSGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    std::vector<distributions::Dirichlet>& weights,
    std::vector<distributions::GaussWish>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool sparse = false,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for the Groups of Mixtures Clustering model
 *         but with diagonal covariance Gaussians.
 *
 * This function implements the Groups of Mixtues Clustering model algorithm
 * as specified by [5], with the additional "sparse" option but with diagonal
 * covariance Gaussians, i.e. this is a Naive-Bayes assumption. The DGMC uses a
 * Generalised Dirichlet prior on the group mixture weights and Normal cluster
 * distributions (With Normal-Gamma priors). This algorithm is similar to a
 * one-level Hierarchical Dirichlet process with Gaussian observations.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions.
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param weights is a vector of distributions over the mixture weights of the
 *         model, for each group of data, J.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          negative diagonal covariance matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnDGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    std::vector<distributions::GDirichlet>& weights,
    std::vector<distributions::NormGamma>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool sparse = false,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


/*! \brief The learning algorithm for the Exponential Groups of Mixtures
 *         Clustering model.
 *
 * This function implements the Exponential Groups of Mixtures Clustering model,
 * with the additional "sparse" option. The Exponential GMC uses a Generalised
 * Dirichlet prior on the group mixture weights, but an Exponential cluster
 * distribution (with a Gamma prior). This algorithm is similar to a
 * one-level Hierarchical Dirichlet process with Exponential observations.
 *
 *  \param X the observation matrices. Vector of N_jxD matrices where N_j is
 *         the number of observations in each group, j, and D is the number
 *         of dimensions. X MUST be in the range [0, inf).
 *  \param qZ is a vector of N_jxK matrices of the variational posterior
 *         approximations to p(z_j|X_j). K is the number of model clusters.
 *         This will always be overwritten to start with one cluster.
 *  \param weights is a vector of distributions over the mixture weights of the
 *         model, for each group of data, J.
 *  \param clusters is a vector of distributions over the cluster parameters
 *         of the model, this will be size K.
 *  \param clusterprior is the prior 'tuning' parameter for the cluster
 *         parameter distributions. This effects how many clusters will be
 *         found.
 *  \param maxclusters is the maximum number of clusters to search for, -1
 *         (default) means no upper bound.
 *  \param sparse flag for enabling the "sparse" updates for the GMC. Some
 *         small amount of accuracy is traded off for a potentially large
 *         speed increase by not updating zero group weight cluster
 *         observation likelihoods. By default this is not enabled.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnEGMC (
    const vMatrixXd& X,
    vMatrixXd& qZ,
    std::vector<distributions::GDirichlet>& weights,
    std::vector<distributions::ExpGamma>& clusters,
    const double clusterprior = PRIORVAL,
    const int maxclusters = -1,
    const bool sparse = false,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


//
// Simultaneous Clustering Models (scluster.cpp)
//


/*! \brief The learning algorithm for the "Simultaneous Clustering Model".
 *
 * This function implements the "Simultaneous Clustering Model" algorithm
 * as specified by [4, 5]. The SCM uses a Generalised Dirichlet prior on the
 * group mixture weights, a Dirichlet prior on the top-level clusters and
 * Gaussian bottom-level cluster distributions for observations (with
 * Gausian-Wishart priors).
 *
 *  \param X the observation matrices. A vector of length J (for each group),
 *         of vectors of length I_j (for each image/document etc) of N_jixD
 *         matrices. Here N_ji is the number of observations in each document,
 *         I_j, in group, j, and D is the number of dimensions.
 *  \param qY the probabilistic label of documents/images to top-level clusters.
 *         It is a vector of length J (for each group), of I_jxT matrices of the
 *         soft assignments of each document/image (i) to a cluster label (t).
 *         It is the variational posterior to p(y_j|Z_j). This is randomly
 *         initialised, and uses the parameter T for max number of classes.
 *  \param qZ the probabilistic label of observations to bottom-level clusters.
 *         It is a vector of length J (for each group), of vectors of length I_j
 *         (for each "document") of N_jixK matrices of the variational posterior
 *         approximations to p(z_ji|X_ji). K is the number of bottom-level
 *         clusters. This will always be overwritten to start with one cluster.
 *  \param weights_j is a vector of distributions over the weights of top-level 
 *         cluster mixtures of the model, for each group of data, J, like the 
 *         GMC.
 *  \param weights_t is a vector of distributions over the weights that 
 *         parameterise the top-level clusters. These also parameterise the 
 *         distribution of the bottom-level cluster weights -- this will be of 
 *         size T* (see parameter T).
 *  \param clusters is a vector of distributions over the segment cluster
 *         parameters of the model, this will be size K.
 *  \param dirprior is the prior 'tuning' parameter for the top-level dirichlet
 *         cluster parameter distributions. This effects how many top-level 
 *         clusters will be found.
 *  \param gausprior is the prior 'tuning' parameter for the bottom-level
 *         Gaussian cluster parameter distributions. This effects how many
 *         clusters will be found.
 *  \param maxT the maximum number of top-level clusters to look for. Usually, 
 *         if maxT is set large, T* < maxT top-level clusters will be found.
 *  \param maxK is the maximum number of bottom level clusters to search for, -1
 *         (default) means no upper bound.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnSCM (
    const vvMatrixXd& X,
    vMatrixXd& qY,
    vvMatrixXd& qZ,
    std::vector<distributions::GDirichlet>& weights_j,
    std::vector<distributions::Dirichlet>& weights_t,
    std::vector<distributions::GaussWish>& clusters,
    const double dirprior = PRIORVAL,
    const double gausprior = PRIORVAL,
    const unsigned int maxT = TRUNC,
    const int maxK = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


//
// Multiple Observation Clustering Models (mcluster.cpp)
//


/*! \brief The learning algorithm for the "Multiple-source Clustering Model".
 *
 * This function implements the "Multiple-source Clustering Model" algorithm as
 * specified by [3 - 5]. This model jointly cluster both "document" level
 * observations, and "word" observations. The MCM uses a Generalised
 * Dirichlet prior on the group mixture weights, Multinomial-Gaussian top-level
 * (document) clusters, and Gaussian bottom-level (word) cluster distributions.
 *
 *  \param W the top-level observation matrices. Vector of length J with N_jxDt 
 *         matrices where N_j is the number of observations in each group, j, 
 *         and Dt is the number of dimensions. J is the total number of groups.
 *  \param X the bottom-level observation matrices. A vector of length J (for 
 *         each group), of vectors of length I_j (for each image or "document") 
 *         of N_jixDb matrices. Here N_ji is the number of observations in each
 *         document, I_j, in group, j, and Db is the number of dimensions.
 *  \param qY the probabilistic label of documents/images to top-level clusters.
 *         It is a vector of length J (for each group), of I_jxT matrices of the
 *         soft assignments of each document/image (i) to a cluster label (t).
 *         It is the variational posterior to p(y_j|Z_j). This is randomly
 *         initialised, and uses the parameter T for max number of classes.
 *  \param qZ the probabilistic label of observations to bottom-level clusters.
 *         It is a vector of length J (for each group), of vectors of length I_j
 *         (for each "document") of N_jixK matrices of the variational posterior
 *         approximations to p(z_ji|X_ji). K is the number of bottom-level
 *         clusters. This will always be overwritten to start with one cluster.
 *  \param weights_j is a vector of distributions over the weights of top-level 
 *         cluster mixtures of the model, for each group of data, J, like the 
 *         GMC.
 *  \param weights_t is a vector of distributions over the weights that 
 *         partially parameterise the top-level clusters. These also 
 *         parameterise the distribution of the bottom-level cluster weights -- 
 *         this will be of size T* (see parameter T).
 *  \param clusters_t is a vector of distributions over the top-level clusters
 *         parameters of the model (corresponding to W), this will be size T*.
 *  \param clusters_k is a vector of distributions over the bottom-level cluster
 *         parameters of the model (corresponding to X), this will be size K.
 *  \param prior_t is the prior 'tuning' parameter for the top-level (Gaussian) 
 *         cluster parameter distributions (W). This effects how many top level
 *         clusters will be found.
 *  \param prior_k is the prior 'tuning' parameter for the bottom-level cluster
 *         parameter distributions (X). This effects how many bottom-level 
 *         clusters will be found.
 *  \param maxT the maximum number of top-level clusters to look for. Usually,
 *         if maxT is set large, T* < maxT top-level clusters will be found.
 *  \param maxK is the maximum number of bottom level clusters to search for, -1
 *         (default) means no upper bound.
 *  \param verbose flag for triggering algorithm status messages. Default is
 *         0 = silent.
 *  \param nthreads sets the number of threads for the clustering algorithm to
 *         use. The default value is automatically determined by OpenMP.
 *  \returns Final free energy
 *  \throws std::logic_error if there are invalid argument calls such as
 *          non-PSD matrix calculations.
 *  \throws std::runtime_error if there are runtime issues with the GMC
 *          algorithm such as negative free energy steps, unexpected empty
 *          clusters etc.
 */
double learnMCM (
    const vMatrixXd& W,
    const vvMatrixXd& X,
    vMatrixXd& qY,
    vvMatrixXd& qZ,
    std::vector<distributions::GDirichlet>& weights_j,
    std::vector<distributions::Dirichlet>& weights_t,
    std::vector<distributions::GaussWish>& clusters_t,
    std::vector<distributions::GaussWish>& clusters_k,
    const double prior_t = PRIORVAL,
    const double prior_k = PRIORVAL,
    const unsigned int maxT = TRUNC,
    const int maxK = -1,
    const bool verbose = false,
    const unsigned int nthreads = omp_get_max_threads()
    );


}
#endif // LIBCLUSTER_H
