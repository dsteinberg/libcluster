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

// I-GMC Classify Mex file entry point
//
//  Inputs:
//      - X, [NxD double] matrix of observations. Essential.
//      - IGMC structure. Essential.
//      - verbose, [logical] (optional) true = verbose output, false = quiet 
//          (default).
//
//  Outputs:
//      - qZ [NxK] double of probability assignments
//      - wj [1xK] double of mixture weights
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool   *verbptr;

    switch (nrhs)
    {
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

    case 1:
    default:
        mexErrMsgTxt("Wrong number of inputs! Need at least X and IGMC!");
    }

    // Map X matlab matrix to eigen matrix
    Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));

    // Make a I-GMC object, qZ and wj matrices
    IGMC igmc = str2igmc(prhs[1]);
    RowVectorXd wj;
    MatrixXd qZ;

    // Call various versions of classifyIGMC depending on the arguments
    mexstreambuf mexout;
    ostream mout(&mexout);
    try
    {
        if (nrhs == 2)
            wj = classifyIGMC(X, igmc, qZ);
        else if (nrhs == 3)
            wj = classifyIGMC(X, igmc, qZ, verbptr[0], mout);
    }
    catch (logic_error e)
        { mexErrMsgTxt(e.what()); }
    catch (runtime_error e)
        { mexErrMsgTxt(e.what()); }

    // Create outputs
    if (nlhs != 2)
        mexErrMsgTxt("Wrong number of outputs.");

    // Cluster probabilities
    plhs[0] = eig2mat(qZ);

    // GMM weights
    plhs[1] =  eig2mat(wj);
}
