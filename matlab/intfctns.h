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

// Interfaces for commonly used functions/classes between the various mex files.

#ifndef INTFCTNS_H
#define INTFCTNS_H

#include <Eigen/Dense>
#include <iostream>
#include "mex.h"
#include "libcluster.h"


//
// Helper classes and structures
//

/*! \brief Mex stream buffer class, prints using mexPrintf() and can be used to
 *         replace the normal cout stream buffer.
 *  \see hijack()
 *  \see restore()
 */
class mexstreambuf : public std::streambuf
{
public:

  /*! \brief Default Constructor */
  mexstreambuf () : coutbak(0) {}

  /*! \brief Hijack the cout stream buffer and replace it with this one. */
  void hijack ();

  /*! \brief Restore the cout stream buffer if it has been hijacked. */
  void restore ();

  /*! \brief Destructor returns cout to its normal state */
  ~mexstreambuf ();

protected:

  virtual std::streamsize xsputn (const char* s, std::streamsize n);

  virtual int overflow (int c = EOF);

  virtual int sync ();

private:

  std::streambuf *coutbak; //!< pointer to normal cout stream buffer.

};


/*! \brief Options structure for setting default, and parsing options from
 *         Matlab to the clustering algorithms.
 */
struct Options
{
public:

  /*! \brief Default Constructor, sets default options */
  Options ()
    : verbose(false),
      sparse(false),
      prior(libcluster::PRIORVAL),
      prior2(libcluster::PRIORVAL),
      trunc(libcluster::TRUNC),
      threads(omp_get_max_threads())
  {}

  /*! \brief Parses an mxArray structure for the relevant fields, if they are
   *         not found, then the defaults are used.
   *
   *  \param optsstruct is a matlab structure array - errors out if its not.
   */
  void parseopts (const mxArray* optsstruct);

  bool verbose;           //!< Verbose output flag
  bool sparse;            //!< Use sparse clustering flag
  double prior;           //!< The cluster prior parameter
  double prior2;          //!< Another cluster prior parameter
  unsigned int trunc;     //!< Truncation level for max number of classes
  unsigned int threads;   //!< Number of threads to use
};


//
// Helper Functions
//

/*! \brief Copy a 2D double Eigen matrix to 2D double mxArray.
 *
 *  \param X eigen matrix.
 *  \returns a pointer to a mxArray with the copied elements.
 */
mxArray* eig2mat (const Eigen::MatrixXd& X);


/*! \brief Map a cell array of matlab matrices to a vector of Eigen matrices.
 *
 *  \param X is a cell array of (double) matrices.
 *  \returns a reference to a vector of mapped eigen double matrices.
 */
libcluster::vMatrixXd cell2vec (const mxArray* X);


/*! \brief Map a cell array of cell arrays of matlab matrices to a vector of
 *         vectors Eigen matrices.
 *
 *  \param X is a cell array of cell arrays of (double) matrices.
 *  \returns a reference to a vector of vectors of mapped eigen double matrices.
 */
libcluster::vvMatrixXd cellcell2vecvec (const mxArray* X);


/*! \brief Copy a vector of Eigen matrices to a cell array of Matlab matrices.
 *
 *  \param X is a vector of (double) matrices.
 *  \returns a cell array of double matrices.
 */
mxArray* vec2cell (const libcluster::vMatrixXd& X);


/*! \brief Copy a vector of vectors of Eigen matrices to a cell array cell
 *         arrays of Matlab matrices.
 *
 *  \param X is a vector of vectors (double) matrices.
 *  \returns a cell array of cell arrays of double matrices.
 */
mxArray* vecvec2cellcell (const libcluster::vvMatrixXd& X);

#endif // INTFCTNS_H
