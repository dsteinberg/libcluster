//
// Includes and symbolic constants
//

#include <stdexcept>
#include <omp.h>
#include "intfctns.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;

//
// Functions
//

// Topic Clustering Models Mex file entry point
//
//  Inputs:
//      - X, {I x [NixD double]} cell of matrices of observations, required.
//      - SS, SuffStat struct, required:
//          SS.K        = scalar double number of clusters.
//          SS.priorval = Prior cluster hyperparameter value.
//          SS.N_k      = {1xK} array of observation counts.
//          SS.ss1      = {1x[?x?]} array of observation suff. stats. no 1.
//          SS.ss2      = {1x[?x?]} array of observation suff. stats. no 2.
//      - SSdocs, {I X SuffStat} cell array of sufficient statistic structures.
//          for each document of data input, required.
//      - T, [integer] truncation level of classes, i.e. max number of classes 
//          to find
//      - alg, [integer] type of algorithm [0=TCM], required.
//      - sparse, [logical] 1=sparse algorithm, 0=original, required.
//      - verbose, [logical] 1=verbose output, 0=quiet, required.
//      - nthreads, [integer] number of threads to use for clustering, optional.
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, {I x [NixK double]} probability of observation n belonging to a
//        document word cluster.
//      - qY, [I x T] probablity of an image i belonging to class t.
//      - SSdocs, {I X SuffStat} cell array of sufficient statistic structures
//          for each document of data input.
//      - SS, SuffStat struct for all of the data in the model.
//      - clparams, [T x K] class parameters (each class is row t)
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   
  // Parse arguments
  if ( (nrhs != 7) && (nrhs != 8) )
    mexErrMsgTxt("Wrong number of inputs!");
  
  if (
        mxGetM(prhs[1]) != 1 
        || mxGetN(prhs[1]) != 1
        || mxIsStruct(prhs[1]) == false
     )
       mexErrMsgTxt("Need a Sufficient Stat. structure!");
  if (
        (mxGetM(prhs[2]) > 1
        && mxGetN(prhs[2]) > 1)
        || mxIsCell(prhs[2]) == false
     )
       mexErrMsgTxt("Need a cell vector of Sufficient Stat. structures!");
  if (
        mxGetM(prhs[3]) != 1 
        || mxGetN(prhs[3]) != 1 
        || mxIsDouble(prhs[3]) == false
     )
       mexErrMsgTxt("T truncation level should be one integer element.");
  if (
        mxGetM(prhs[4]) != 1 
        || mxGetN(prhs[4]) != 1 
        || mxIsDouble(prhs[4]) == false
     )
       mexErrMsgTxt("algval enum should be one integer element.");
  if (
        mxGetM(prhs[5]) != 1 
        || mxGetN(prhs[5]) != 1 
        || mxIsLogical(prhs[5]) == false
     )
       mexErrMsgTxt("sparse flag should be one logical element.");
  if (
        mxGetM(prhs[6]) != 1 
        || mxGetN(prhs[6]) != 1 
        || mxIsLogical(prhs[6]) == false
     )
       mexErrMsgTxt("verbose flag should be one logical element.");
 
  bool          *verbptr  = (bool*) mxGetPr(prhs[6]), 
                *sparsptr = (bool*) mxGetPr(prhs[5]);
  double        *algptr   = (double*) mxGetPr(prhs[4]),
                *Tptr     = (double*) mxGetPr(prhs[3]), 
                *thrptr;
  const mxArray *SSptr     = prhs[1], 
                *SSdocsptr = prhs[2];
                
  if  (nrhs == 8)
  {
    if (
          mxGetM(prhs[7]) != 1 
          || mxGetN(prhs[7]) != 1 
          || mxIsDouble(prhs[7]) == false
       )
        mexErrMsgTxt("nthreads should be one unsigned integer element.");
    thrptr = (double*) mxGetPr(prhs[7]);
  }
    
  // Number of groups and Dimensions
  int I = mxGetN(prhs[0]) > mxGetM(prhs[0]) 
          ? mxGetN(prhs[0]) : mxGetM(prhs[0]);
  int D = mxGetN(mxGetCell(prhs[0], 0));

  // Map X matlab cells to vector of eigen matrices.
  vector<MatrixXd> X;
  vector<int> Ni;
  for (int i = 0; i < I; ++i)
  {
      Ni.push_back(mxGetM(mxGetCell(prhs[0], i))); 
      Map<MatrixXd> Xi(mxGetPr(mxGetCell(prhs[0], i)), Ni[i], D);
      X.push_back(Xi);
  }

  // Make SuffStat objects, qZ and qY matrices and Free energy double
  double F;
  SuffStat SS = str2SS(SSptr);
  vector<SuffStat> SSdocs;
  vector<MatrixXd> qZ;
  MatrixXd qY, clparams;

  for (int i = 0; i < I; ++i)
    SSdocs.push_back(str2SS(mxGetCell(SSdocsptr, i)));

  // redirect cout
  mexstreambuf mexout;
  streambuf *coutbak; 
  coutbak = cout.rdbuf();
  cout.rdbuf(&mexout);
  
  // Call various versions of clustering algorithms depending on the arguments
  try
  {
    switch ((int) algptr[0])
    {
      case TCM:
        F = learnTCM(X, qY, qZ, SSdocs, SS, clparams, (unsigned int) Tptr[0], 
                     verbptr[0]);
        break;
              
      default:
        throw logic_error("Wrong algorithm type specified!");
    }  
  }
  catch (logic_error e) 
    { mexErrMsgTxt(e.what()); }
  catch (runtime_error e)
    { mexErrMsgTxt(e.what()); }

  // Restore cout
  cout.rdbuf(coutbak);

  // Create outputs  
  if (nlhs != 6) 
      mexErrMsgTxt("Wrong number of outputs.");

  // Free energy
  plhs[0] = mxCreateDoubleScalar(F);

  // Copy eigen qZ and SSdocs to mxArray qZ and SSdocs
  mxArray *qZmx = mxCreateCellMatrix(1, I);
  mxArray *SSmx = mxCreateCellMatrix(1, I);

  for (int i = 0; i < I; ++i)
  {       
      mxSetCell(qZmx, i, eig2mat(qZ[i]));    // copy qZ
      mxSetCell(SSmx, i, SS2str(SSdocs[i])); // copy SSdocs
  } 

  plhs[1] = eig2mat(qY);
  plhs[2] = qZmx;
  plhs[3] = SSmx;
  plhs[4] = SS2str(SS);   // Copy SS to a matlab compatible structure
  plhs[5] = eig2mat(clparams);
  
}
