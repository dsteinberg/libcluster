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

#include "mintfctns.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;


//
// Class and Struct definitions
//

void mexstreambuf::hijack ()
{
  // redirect cout
  this->coutbak = cout.rdbuf();
  cout.rdbuf(this);
}


void mexstreambuf::restore ()
{
  if (this->coutbak != 0)
  {
    cout.rdbuf(this->coutbak);
    this->coutbak = 0;
  }
}


mexstreambuf::~mexstreambuf ()
{
  // make sure cout restored in case it was hijacked and not retored explicitly
  this->restore();
}


streamsize mexstreambuf::xsputn (const char *s, streamsize n)
{
  mexPrintf("%.*s", n, s);
  return n;
}


int mexstreambuf::overflow (int c)
{
  if (c != EOF)
    mexPrintf("%.1s", &c);
  return 1;
}


int mexstreambuf::sync ()
{
  if (pbase() == pptr()) // If the buffers is syncronised, do nothing
  {
    mexEvalString("drawnow;");  // print to the console now
    return 1;
  }
  else // Else try and write the contents of the buffer
  {
    int cnwrite = mexPrintf(pbase());
    mexEvalString("drawnow;");  // print to the console now
    return cnwrite ? 0 : 1;
  }
}


void Options::parseopts (const mxArray* optsstruct)
{
  if (mxIsStruct(optsstruct) == false)
    mexErrMsgTxt("optsstruct must be a matlab structure!");

  // Look for the "verbose" field
  mxArray* vbfield = mxGetField(optsstruct, 0, "verbose");
  if (vbfield != 0)
    this->verbose = (bool) *mxGetPr(vbfield);

  // Look for the "sparse" field
  mxArray* spfield = mxGetField(optsstruct, 0, "sparse");
  if (spfield != 0)
    this->sparse = (bool) *mxGetPr(spfield);

  // Look for the "threads" field
  mxArray* thfield = mxGetField(optsstruct, 0, "threads");
  if (thfield != 0)
    this->threads = (unsigned int) *mxGetPr(thfield);

  // Look for the "prior" field
  mxArray* prfield = mxGetField(optsstruct, 0, "prior");
  if (prfield != 0)
    this->prior = (double) *mxGetPr(prfield);

  // Look for the "prior2" field
  mxArray* prfield2 = mxGetField(optsstruct, 0, "prior2");
  if (prfield2 != 0)
    this->prior2 = (double) *mxGetPr(prfield2);

  // Look for the "trunc" field
  mxArray* trfield = mxGetField(optsstruct, 0, "trunc");
  if (trfield != 0)
    this->trunc = (unsigned int) *mxGetPr(trfield);
}


//
// Functions
//

mxArray* eig2mat(const MatrixXd& X)
{
  const unsigned int Ntot = X.cols() * X.rows();

  // Create a new mxArray to return
  mxArray *rX = mxCreateDoubleMatrix(X.rows(), X.cols(), mxREAL);

  // Create pointers to the underlying col major data structures.
  const double *Xptr = X.data();
  double *rXptr = mxGetPr(rX);

  // Copy
  for (unsigned int n = 0; n < Ntot; ++n)
    rXptr[n] = Xptr[n];

  return rX; // return copy
}


vMatrixXd cell2vec(const mxArray* X)
{
  // Initial cell array checking
  if (mxIsCell(X) == false)
    mexErrMsgTxt("X must be a matlab cell array!");
  if ( (mxGetN(X) > 1) && (mxGetM(X) > 1) )
    mexErrMsgTxt("X must be either a {1xJ} or {Jx1} array!");
  if ( (mxGetN(X) == 0) || (mxGetM(X) == 0) )
    mexErrMsgTxt("X is empty, need some data!");

  const unsigned int J = mxGetN(X) > mxGetM(X) ? mxGetN(X) : mxGetM(X);
  const unsigned int D = mxGetN(mxGetCell(X, 0));

  // More detailed array checking
  for (unsigned int j = 0; j < J; ++j)
  {
    if (mxIsDouble(mxGetCell(X, j)) == false)
      mexErrMsgTxt("X must be all array double precision matrices!");
    if (mxGetN(mxGetCell(X, j)) != D)
      mexErrMsgTxt("X must contain matrices with the same number of columns!");
  }

  // Map each matrix in the array
  vMatrixXd rX;
  for (unsigned int j = 0; j < J; ++j)
    rX.push_back(Map<MatrixXd>(mxGetPr(mxGetCell(X, j)),
                              mxGetM(mxGetCell(X, j)), D));

  return rX;
}


vvMatrixXd cellcell2vecvec (const mxArray* X)
{
  // Initial cell array checking
  if (mxIsCell(X) == false)
    mexErrMsgTxt("X must be a matlab cell array!");
  if ( (mxGetN(X) > 1) && (mxGetM(X) > 1) )
    mexErrMsgTxt("X must be either a {1xJ} or {Jx1} array!");
  if ( (mxGetN(X) == 0) || (mxGetM(X) == 0) )
    mexErrMsgTxt("X is empty, need some data!");

  const unsigned int J = mxGetN(X) > mxGetM(X) ? mxGetN(X) : mxGetM(X);

  vvMatrixXd rX;
  for (unsigned int j = 0; j < J; ++j)
    rX.push_back(cell2vec(mxGetCell(X, j)));

  return rX;
}


mxArray* vec2cell (const vMatrixXd& X)
{
  const unsigned int J = X.size();

  // Allocate a new Cell array
  mxArray *rX = mxCreateCellMatrix(1, J);

  // Copy Matrices
  for (unsigned int j = 0; j < J; ++j)
    mxSetCell(rX, j, eig2mat(X[j]));

  return rX; // return copy
}


mxArray* vecvec2cellcell (const vvMatrixXd& X)
{
  const unsigned int J = X.size();

  // Allocate a new Cell array
  mxArray *rX = mxCreateCellMatrix(1, J);

  // Copy Cell Arrays
  for (unsigned int j = 0; j < J; ++j)
    mxSetCell(rX, j, vec2cell(X[j]));

  return rX; // return copy
}
