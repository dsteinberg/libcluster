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

// Mixture models for clustering Mex file entry point
//
//  Inputs:
//      - X, [NxD double] matrix of observations, required
//      - SS, SuffStat struct, required:
//          SS.K        = scalar double number of clusters.
//          SS.priorval = Prior cluster hyperparameter value.
//          SS.N_k      = {1xK} array of observation counts.
//          SS.ss1      = {1x[?x?]} array of observation suff. stats. no 1.
//          SS.ss2      = {1x[?x?]} array of observation suff. stats. no 2.
//      - alg, [integer] the type of algorithm [VDP=0, BGMM=1, DGMM=2, BEMM=3], 
//          required.
//      - verbose, [logical] 1=verbose output, 0=quiet, required.
//      - nthreads, [integer] number of threads to use for clustering, optional.
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, [NxK double] probability of observation n belonging to cluster k.
//      - SS, SuffStat struct for all of the data in the model.
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  // Parse arguments
  if ( (nrhs != 4) && (nrhs != 5) )
    mexErrMsgTxt("Wrong number of inputs!");
  
  if (
        mxGetM(prhs[1]) != 1 
        || mxGetN(prhs[1]) != 1
        || mxIsStruct(prhs[1]) == false
     )
       mexErrMsgTxt("Need a Sufficient Stat. structure!");
  if (
        mxGetM(prhs[2]) != 1 
        || mxGetN(prhs[2]) != 1 
        || mxIsDouble(prhs[2]) == false
     )
       mexErrMsgTxt("algval flag should be one integer element.");
  if (
        mxGetM(prhs[3]) != 1 
        || mxGetN(prhs[3]) != 1 
        || mxIsLogical(prhs[3]) == false
     )
       mexErrMsgTxt("verbose should be one logical element.");
    
  bool    *verbptr = (bool*) mxGetPr(prhs[3]);
  double  *algptr  = (double*) mxGetPr(prhs[2]),
          *thrptr;
  const mxArray *SSptr = prhs[1];
  
  if (nrhs == 5)
  { 
    if (
          mxGetM(prhs[4]) != 1 
          || mxGetN(prhs[4]) != 1
          || mxIsDouble(prhs[4]) == false
       )
         mexErrMsgTxt("nthreads should be one unsigned integer element.");
    thrptr = (double*) mxGetPr(prhs[4]);
  }
  
  // Map X matlab matrix to eigen matrix
  Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));
  
  // Make a SuffStat object, qZ matrix and Free energy double
  SuffStat SS = str2SS(SSptr);
  MatrixXd qZ;
  double F;
  
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
      case VDP:
        if (nrhs == 4)
          F = learnVDP(X, qZ, SS, verbptr[0]);
        else
          F = learnVDP(X, qZ, SS, verbptr[0], (unsigned int) thrptr[0]);
        break;
        
      case BGMM:
        if (nrhs == 4)
          F = learnBGMM(X, qZ, SS, verbptr[0]);
        else
          F = learnBGMM(X, qZ, SS, verbptr[0], (unsigned int) thrptr[0]);
        break;
        
      case DGMM:
        if (nrhs == 4)
          F = learnDGMM(X, qZ, SS, verbptr[0]);
        else
          F = learnDGMM(X, qZ, SS, verbptr[0], (unsigned int) thrptr[0]);
        break;
        
      case BEMM:
        if (nrhs == 4)
          F = learnBEMM(X, qZ, SS, verbptr[0]);
        else
          F = learnBEMM(X, qZ, SS, verbptr[0], (unsigned int) thrptr[0]);
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
  if (nlhs != 3) 
      mexErrMsgTxt("Wrong number of outputs.");
  
  // Free energy
  plhs[0] = mxCreateDoubleScalar(F);
  
  // Copy eigen qz to mxArray qz   
  plhs[1] = eig2mat(qZ);
  
  // Copy SS to a matlab compatible structure
  plhs[2] = SS2str(SS);
}
