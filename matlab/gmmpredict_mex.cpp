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

// GMM Predict Mex file entry point
//
//  Inputs:
//      - X, [NxD double] matrix of observations. Essential.
//  	- A GMM struct of the form:
//          gmm.K     = scalar double number of clusters
//          gmm.w     = {1xK} array of scalar weights
//          gmm.mu    = {1x[1xD]} array of means
//          gmm.sigma = {1x[DxD]} array of covariances
//
//  Outputs:
//      - pX, [Nx1 double] probability of observation according to the GMM.
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Some input parsing
    if (nrhs != 2)
        mexErrMsgTxt("Wrong number of inputs.");
    
    // Map X matlab matrix to eigen matrix
    Map<MatrixXd> X(mxGetPr(prhs[0]), mxGetM(prhs[0]), mxGetN(prhs[0]));

    // Convert the GMM object, and predict!
    GMM gmm;
    VectorXd pX;
    try 
    { 
        gmm = str2gmm(prhs[1]);
        pX = predict(X, gmm);
    }
    catch (logic_error e) 
        {  mexErrMsgTxt(e.what()); }
    
    // Create outputs  
    if (nlhs != 1) 
        mexErrMsgTxt("Wrong number of outputs.");
    
    // Get various dimensionalities etc
    int N = X.rows();
    
    // Parse back result
    plhs[0] = eig2mat(pX);
}
