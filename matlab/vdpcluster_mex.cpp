//
// Includes and symbolic constants
//

#include <stdexcept>
#include "intfctns.h"

#define VERBDEF     0
#define CWIDTHDEF   0.01f

//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;

//
// Functions
//

// VDP Cluster Mex file entry point
//
//  Inputs:
//      - X, [NxD double] matrix of observations. Essential.
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
    case 3:
        if (
                mxGetM(prhs[2]) != 1 
                || mxGetN(prhs[2]) != 1
                || mxIsDouble(prhs[2]) == false
           )
            mexErrMsgTxt("Clusterwidth should be one double element.");
        clstwptr = (double*)mxGetPr(prhs[2]);
        
    case 2:
        if (
                mxGetM(prhs[1]) != 1 
                || mxGetN(prhs[1]) != 1 
                || mxIsLogical(prhs[1]) == false
           )
            mexErrMsgTxt("Verbose should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[1]);
        
    case 1:
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
    
    // Call various versions of learnvdp depending on the arguments
    mexstreambuf mexout;
    ostream mout(&mexout);
    try
    {
        if (nrhs == 1)
            F = learnVDP(X, qZ, gmm, VERBDEF, CWIDTHDEF, mout);
        else if (nrhs == 2)
            F = learnVDP(X, qZ, gmm, verbptr[0], CWIDTHDEF, mout);
        else if (nrhs == 3)
            F = learnVDP(X, qZ, gmm, verbptr[0], clstwptr[0], mout);
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
