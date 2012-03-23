//
// Includes and symbolic constants
//

#include <stdexcept>
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
//      - X, [NxD double] matrix of observations, Required
//      - alg, [integer] the type of algorithm [0=VDP, 1=Bayesian SS], Required.
//      - diagcov, [logical] (optional) 1=diagonal cov., 0=full cov. (default).
//      - verbose, [logical] (optional) 1=verbose output, 0=quiet (default).
//      - clustwidth, [double] (optional) cluster prior width (default = 1e-5).
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, [NxK double] probability of observation n belonging to cluster k.
//      - SS, SuffStat struct:
//          SS.K        = scalar double number of clusters
//          SS.priorval = Prior cluster hyperparameter value.
//          SS.N_k      = {1xK} array of observation counts
//          SS.ss1      = {1x[?x?]} array of observation suff. stats. no 1.
//          SS.ss2      = {1x[?x?]} array of observation suff. stats. no 2.
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool   *verbptr, *diagptr;
    double *clstwptr;
    
    switch (nrhs)
    {
    case 5:
        if (
                mxGetM(prhs[4]) != 1 
                || mxGetN(prhs[4]) != 1
                || mxIsDouble(prhs[4]) == false
           )
            mexErrMsgTxt("clustwidth should be one double element.");
        clstwptr = (double*)mxGetPr(prhs[4]);
        
    case 4:
        if (
                mxGetM(prhs[3]) != 1 
                || mxGetN(prhs[3]) != 1 
                || mxIsLogical(prhs[3]) == false
           )
            mexErrMsgTxt("verbose should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[3]);
        
    case 3:
        if (
                mxGetM(prhs[2]) != 1 
                || mxGetN(prhs[2]) != 1 
                || mxIsLogical(prhs[2]) == false
           )
            mexErrMsgTxt("diagov should be one logical element.");
        diagptr = (bool*)mxGetPr(prhs[2]);
        
    case 2:
        break;
        
    default:
        mexErrMsgTxt("Wrong number of inputs!");
    }
    
    // Map X matlab matrix to eigen matrix
    Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));
    
    // Make a SuffStat object, qZ matrix and Free energy double
    SuffStat SS;
    if (nrhs == 4)
      SS = SuffStat(clstwptr[0]);
    MatrixXd qZ;
    double F;
    
    // Call various versions of clustering algorithms depending on the arguments
    double *algptr = (double*)mxGetPr(prhs[1]);
    int algval = (int) algptr[0];
    mexstreambuf mexout;
    ostream mout(&mexout);
    try
    {   
        if (algval == VDP)
        {
            if (nrhs == 2)
                F = learnVDP(X, qZ, SS, DIAGDEF, VERBDEF, mout);
            else if (nrhs == 3)
                F = learnVDP(X, qZ, SS, diagptr[0], VERBDEF, mout);
            else if (nrhs >= 4)
                F = learnVDP(X, qZ, SS, diagptr[0], verbptr[0], mout);
        }
        else if (algval == BGMM)
        {
            if (nrhs == 2)
                F = learnGMM(X, qZ, SS, DIAGDEF, VERBDEF, mout);
            else if (nrhs == 3)
                F = learnGMM(X, qZ, SS, diagptr[0], VERBDEF, mout);
            else if (nrhs >= 4)
                F = learnGMM(X, qZ, SS, diagptr[0], verbptr[0], mout);
        }
        else
            throw logic_error("Wrong algorithm type specified!");
    }
    catch (logic_error e) 
        { mexErrMsgTxt(e.what()); }
    catch (runtime_error e) 
        { mexErrMsgTxt(e.what()); }
    
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
