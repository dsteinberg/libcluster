//
// Includes and symbolic constants
//

#include <stdexcept>
#include "intfctns.h"

#define SPARSDEF    0
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

// GMC Cluster Mex file entry point
//
//  Inputs:
//      - X, {J x [NjxD double]} cell of matrices of observations. Essential.
//      - sparse, [logical] (optional) 1=sparse algorithm, 0=original (default).
//      - verbose, [logical] (optional) 1=verbose output, 0=quiet (default).
//      - clustwidth, [double] (opition) cluster prior width (default = 0.01).
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, {J x [NjxK double]} probability of observation n belonging to a
//        group cluster.
//      - wj, {J x [1xK double]} weights of group clusters.
//      - GMM struct:
//          gmm.K     = scalar double number of clusters
//          gmm.w     = {1xK} array of scalar weights
//          gmm.mu    = {1x[1xD]} array of means
//          gmm.sigma = {1x[DxD]} array of covariances
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool   *verbptr, *sparsptr;
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
            mexErrMsgTxt("Verbose flag should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[2]);
    
    case 2:
        if (
                mxGetM(prhs[1]) != 1 
                || mxGetN(prhs[1]) != 1 
                || mxIsLogical(prhs[1]) == false
           )
            mexErrMsgTxt("Sparse flag should be one logical element.");
        sparsptr = (bool*)mxGetPr(prhs[1]);
        
    case 1:
        break;
        
    default:
        mexErrMsgTxt("Wrong number of inputs!");
    }
    
    // Number of groups and Dimensions
    int J = mxGetN(prhs[0]) > mxGetM(prhs[0]) 
            ? mxGetN(prhs[0]) : mxGetM(prhs[0]);
    int D = mxGetN(mxGetCell(prhs[0], 0));
    
    // Map X matlab cells to vector of eigen matrices. Make qZ.
    vector<MatrixXd> X, qZ;
    MatrixXd Xj;
    vector<int> Nj;
    for (int j = 0; j < J; ++j)
    {
        Nj.push_back(mxGetM(mxGetCell(prhs[0], j))); 
        Map<MatrixXd>  Xj(mxGetPr(mxGetCell(prhs[0], j)), Nj[j], D);
        X.push_back(Xj);
    }
    
    // Make a GMM object, qZ matrix and Free energy double
    GMM gmm;
    double F;
    vector<RowVectorXd> wj;
    
    // Call various versions of learnvdp depending on the arguments
    mexstreambuf mexout;
    ostream mout(&mexout);
    try
    {
        if (nrhs == 1)
            F = learnGMC(X, qZ, wj, gmm, SPARSDEF, VERBDEF, CWIDTHDEF, mout);
        else if (nrhs == 2)
            F = learnGMC(X, qZ, wj, gmm, sparsptr[0], VERBDEF, CWIDTHDEF, mout);
        else if (nrhs == 3)
            F = learnGMC(X,qZ,wj,gmm,sparsptr[0],verbptr[0],CWIDTHDEF,mout);
        else if (nrhs == 4)
            F = learnGMC(X,qZ,wj,gmm,sparsptr[0],verbptr[0],clstwptr[0],mout);
    }
    catch (logic_error e) 
        { mexErrMsgTxt(e.what()); }
    catch (runtime_error e)
        { mexErrMsgTxt(e.what()); }
    
    // Create outputs  
    if (nlhs != 4) 
        mexErrMsgTxt("Wrong number of outputs.");
    
    // Free energy
    plhs[0] = mxCreateDoubleScalar(F);
    
    // Copy eigen qZ and wj to mxArray qZ and wj
    mxArray *qZmx = mxCreateCellMatrix(1, J);
    mxArray *wmx  = mxCreateCellMatrix(1, J);
    
    int K = gmm.getK();
    for (int j = 0; j < J; ++j)
    {       
        mxSetCell(qZmx, j, eig2mat(qZ[j])); // copy qZ
        mxSetCell(wmx, j, eig2mat(wj[j]));  // copy wj
    } 
    
    plhs[1] = qZmx;
    plhs[2] = wmx;
    
    // Copy GMM to a matlab compatible structure
    plhs[3] = gmm2str(gmm);
}
