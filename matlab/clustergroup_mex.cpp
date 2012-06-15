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

// Group Mixture Models for clustering Mex file entry point
//
//  Inputs:
//      - X, {J x [NjxD double]} cell of matrices of observations, required.
//      - SS, SuffStat struct, required:
//          SS.K        = scalar double number of clusters.
//          SS.priorval = Prior cluster hyperparameter value.
//          SS.N_k      = {1xK} array of observation counts.
//          SS.ss1      = {1x[?x?]} array of observation suff. stats. no 1.
//          SS.ss2      = {1x[?x?]} array of observation suff. stats. no 2.
//      - SSgroup, {J X SuffStat} cell array of sufficient statistic structures.
//          for each group of data input, required.
//      - alg, [integer] type of algorithm [GMC=0, SGMC=1, DGMC=2, EGMC=3 ], 
//          required.
//      - sparse, [logical] 1=sparse algorithm, 0=original, required.
//      - verbose, [logical] 1=verbose output, 0=quiet, required.
//      - nthreads, [integer] number of threads to use for clustering, optional.
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, {J x [NjxK double]} probability of observation n belonging to a
//        group cluster.
//      - SSgroup, {J X SuffStat} cell array of sufficient statistic structures
//          for each group of data input.
//      - SS, SuffStat struct for all of the data in the model.
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   
  // Parse arguments
  if ( (nrhs != 6) && (nrhs != 7) )
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
       mexErrMsgTxt("algval flag should be one integer element.");
  if (
        mxGetM(prhs[4]) != 1 
        || mxGetN(prhs[4]) != 1 
        || mxIsLogical(prhs[4]) == false
     )
       mexErrMsgTxt("sparse flag should be one logical element.");
  if (
        mxGetM(prhs[5]) != 1 
        || mxGetN(prhs[5]) != 1 
        || mxIsLogical(prhs[5]) == false
     )
       mexErrMsgTxt("verbose flag should be one logical element.");
 
  bool          *verbptr  = (bool*) mxGetPr(prhs[5]), 
                *sparsptr = (bool*) mxGetPr(prhs[4]);
  double        *algptr   = (double*) mxGetPr(prhs[3]),
                *thrptr;
  const mxArray *SSptr      = prhs[1], 
                *SSgroupptr = prhs[2];
                
  if  (nrhs == 7)
  {
    if (
          mxGetM(prhs[6]) != 1 
          || mxGetN(prhs[6]) != 1 
          || mxIsDouble(prhs[6]) == false
       )
        mexErrMsgTxt("nthreads should be one unsigned integer element.");
    thrptr = (double*) mxGetPr(prhs[6]);
  }
    
  // Number of groups and Dimensions
  int J = mxGetN(prhs[0]) > mxGetM(prhs[0]) 
          ? mxGetN(prhs[0]) : mxGetM(prhs[0]);
  int D = mxGetN(mxGetCell(prhs[0], 0));

  // Map X matlab cells to vector of eigen matrices.
  vector<MatrixXd> X;
  vector<int> Nj;
  for (int j = 0; j < J; ++j)
  {
      Nj.push_back(mxGetM(mxGetCell(prhs[0], j))); 
      Map<MatrixXd> Xj(mxGetPr(mxGetCell(prhs[0], j)), Nj[j], D);
      X.push_back(Xj);
  }

  // Make SuffStat objects, qZ matrix and Free energy double
  double F;
  SuffStat SS = str2SS(SSptr);
  vector<SuffStat> SSgroup;
  vector<MatrixXd> qZ;

  for (int j = 0; j < J; ++j)
    SSgroup.push_back(str2SS(mxGetCell(SSgroupptr, j)));

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
      case GMC:
        if (nrhs == 6)
          F = learnGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0]);
        else
          F = learnGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0], 
                       (unsigned int) thrptr[0]);
        break;
                       
      case SGMC:
        if (nrhs == 6)
          F = learnSGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0]);
        else
          F = learnSGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0], 
                        (unsigned int) thrptr[0]);
        break;
                                
      case DGMC:
        if (nrhs == 6)
          F = learnDGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0]);
        else
          F = learnDGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0], 
                        (unsigned int) thrptr[0]);
        break;
                                
      case EGMC:
        if (nrhs == 6)
          F = learnEGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0]);
        else
          F = learnEGMC(X, qZ, SSgroup, SS, sparsptr[0], verbptr[0], 
                        (unsigned int) thrptr[0]);
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
  if (nlhs != 4) 
      mexErrMsgTxt("Wrong number of outputs.");

  // Free energy
  plhs[0] = mxCreateDoubleScalar(F);

  // Copy eigen qZ and SSgroups to mxArray qZ and SSgroups
  mxArray *qZmx = mxCreateCellMatrix(1, J);
  mxArray *SSmx = mxCreateCellMatrix(1, J);

  for (int j = 0; j < J; ++j)
  {       
      mxSetCell(qZmx, j, eig2mat(qZ[j]));     // copy qZ
      mxSetCell(SSmx, j, SS2str(SSgroup[j])); // copy SSgroups
  } 

  plhs[1] = qZmx;
  plhs[2] = SSmx;

  // Copy SS to a matlab compatible structure
  plhs[3] = SS2str(SS);
}
