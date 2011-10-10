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
//      - alg, [integer] the type of algorithm [0=VDP, 1=Bayesian GMM], Required
//      - verbose, [logical] (optional) 1=verbose output, 0=quiet (default).
//      - clustwidth, [double] (opition) cluster prior width (default = 0.01).
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, [NxK double] probability of observation n belonging to cluster k.
//      - GMM struct:
//          gmm.K     = scalar double number of clusters
//          gmm.w     = {1xK} array of scalar weights
//          gmm.mu    = {1x[1xD]} array of means
//          gmm.sigma = {1x[DxD]} array of covariances
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool   *verbptr;
    double *clstwptr;
    
    switch (nrhs)
    {
    case 4:
        if (
                mxGetM(prhs[3]) != 1 
                || mxGetN(prhs[3]) != 1
                || mxIsDouble(prhs[3]) == false
           )
            mexErrMsgTxt("Clusterwidth should be one double element.");
        clstwptr = (double*)mxGetPr(prhs[3]);
        
    case 3:
        if (
                mxGetM(prhs[2]) != 1 
                || mxGetN(prhs[2]) != 1 
                || mxIsLogical(prhs[2]) == false
           )
            mexErrMsgTxt("Verbose should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[2]);
        
    case 2:
        break;
        
    default:
        mexErrMsgTxt("Wrong number of inputs!");
    }
    
    // Map X matlab matrix to eigen matrix
    Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));
    
    // Make a GMM object, qZ matrix and Free energy double
    GMM gmm;
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
                F = learnVDP(X, qZ, gmm, VERBDEF, CWIDTHDEF, mout);
            else if (nrhs == 3)
                F = learnVDP(X, qZ, gmm, verbptr[0], CWIDTHDEF, mout);
            else if (nrhs == 4)
                F = learnVDP(X, qZ, gmm, verbptr[0], clstwptr[0], mout);
        }
        else if (algval == BGMM)
        {
            if (nrhs == 2)
                F = learnGMM(X, qZ, gmm, VERBDEF, CWIDTHDEF, mout);
            else if (nrhs == 3)
                F = learnGMM(X, qZ, gmm, verbptr[0], CWIDTHDEF, mout);
            else if (nrhs == 4)
                F = learnGMM(X, qZ, gmm, verbptr[0], clstwptr[0], mout);
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
    
    // Copy GMM to a matlab compatible structure
    plhs[2] = gmm2str(gmm);
}
