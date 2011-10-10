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

// I-GMC Cluster Mex file entry point
//
//  Inputs:
//      - X, [NxD double] matrix of observations. Essential.
//      - IGMC structure. Essential.
//      - verbose, [logical] (optional) true = verbose output, false = quiet 
//          (default).
//
//  Outputs:
//      - F, [double] final free energy.
//      - IGMC struct:
//          gmm.K     = scalar double number of clusters
//          gmm.w     = {1xK} array of scalar weights
//          gmm.mu    = {1x[1xD]} array of means
//          gmm.sigma = {1x[DxD]} array of covariances
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool* verbptr;

    switch (nrhs)
    {
    case 3:
        if (
                mxGetM(prhs[2]) != 1
                || mxGetN(prhs[2]) != 1
                || mxIsLogical(prhs[2]) == false
           )
            mexErrMsgTxt("verbose should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[2]);

    case 2:
        break;
        
    case 1:
    default:
        mexErrMsgTxt("Wrong number of inputs! Need X and IGMC!");
    }

    // Map X matlab matrix to eigen matrix
    Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));

    // Make a I-GMC object and Free energy double
    IGMC igmc = str2igmc(prhs[1]);
    bool ischanged;

    // Call various versions of learnIGMC depending on the arguments
    mexstreambuf mexout;
    ostream mout(&mexout);
    try
    {
        if (nrhs == 2)
            ischanged = learnIGMC(X, igmc);
        else if (nrhs == 3)
            ischanged = learnIGMC(X, igmc, verbptr[0], mout);
    } 
    catch (logic_error e)
        { mexErrMsgTxt(e.what()); }
    catch (runtime_error e)
        { mexErrMsgTxt(e.what()); }

    // Create outputs
    if (nlhs != 2)
        mexErrMsgTxt("Wrong number of outputs.");

    // Copy I-GMC to a matlab compatible structure
    plhs[0] = igmc2str(igmc);
    
    // Free energy
    plhs[1] = mxCreateDoubleScalar(ischanged);
}
