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

// Group Mixture Models for clustering Mex file entry point
//
//  Inputs:
//      - X, {J x [NjxD double]} cell of matrices of observations, Required
//      - alg, [integer] the type of algorithm [2=GMC, 3=SGMC], Required
//      - sparse, [logical] (optional) 1=sparse algorithm, 0=original (default).
//      - diagcov, [logical] (optional) 1=diagonal cov., 0=full cov. (default).
//      - verbose, [logical] (optional) 1=verbose output, 0=quiet (default).
//      - SS, SuffStat struct (optional):
//          SS.K        = scalar double number of clusters
//          SS.priorval = Prior cluster hyperparameter value.
//          SS.N_k      = {1xK} array of observation counts
//          SS.ss1      = {1x[?x?]} array of observation suff. stats. no 1.
//          SS.ss2      = {1x[?x?]} array of observation suff. stats. no 2.
//      - SSgroup, {J X SuffStat} cell array of sufficient statistic structures
//          for each group of data input (optional).
//
//  Outputs:
//      - F, [double] final free energy.
//      - qZ, {J x [NjxK double]} probability of observation n belonging to a
//        group cluster.
//      - wj, {J x [1xK double]} weights of group clusters.
//      - SSgroup, {J X SuffStat} cell array of sufficient statistic structures
//          for each group of data input.
//      - SS, SuffStat struct for all of the data in the model.
//
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Parse arguments
    bool *verbptr, *sparsptr, *diagptr;
    const mxArray *SSptr, *SSgroupptr;
    
    switch (nrhs)
    {
    case 7:
        if (
                (mxGetM(prhs[6]) > 1
                && mxGetN(prhs[6]) > 1)
                || mxIsCell(prhs[6]) == false
           )
             mexErrMsgTxt("Need a cell vector of Sufficient Stat. structures!");
        SSgroupptr = prhs[6];

    case 6:
        if (
                mxGetM(prhs[5]) != 1 
                || mxGetN(prhs[5]) != 1
                || mxIsStruct(prhs[5]) == false
           )
             mexErrMsgTxt("Need a Sufficient Stat. structure!");
        SSptr = prhs[5];
    
    case 5:
        if (
                mxGetM(prhs[4]) != 1 
                || mxGetN(prhs[4]) != 1 
                || mxIsLogical(prhs[4]) == false
           )
             mexErrMsgTxt("verbose flag should be one logical element.");
        verbptr = (bool*)mxGetPr(prhs[4]);
    
    case 4:
        if (
                mxGetM(prhs[3]) != 1 
                || mxGetN(prhs[3]) != 1 
                || mxIsLogical(prhs[3]) == false
           )
            mexErrMsgTxt("diagov should be one logical element.");
        diagptr = (bool*)mxGetPr(prhs[3]);
    
    case 3:
        if (
                mxGetM(prhs[2]) != 1 
                || mxGetN(prhs[2]) != 1 
                || mxIsLogical(prhs[2]) == false
           )
             mexErrMsgTxt("sparse flag should be one logical element.");
        sparsptr = (bool*)mxGetPr(prhs[2]);
        
    case 2:
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
    vector<int> Nj;
    for (int j = 0; j < J; ++j)
    {
        Nj.push_back(mxGetM(mxGetCell(prhs[0], j))); 
        Map<MatrixXd> Xj(mxGetPr(mxGetCell(prhs[0], j)), Nj[j], D);
        X.push_back(Xj);
    }
    
    // Make SuffStat objects, qZ matrix and Free energy double
    double F;
    SuffStat SS;
    vector<SuffStat> SSgroup;
    if (nrhs >= 6)
    {
        SS = str2SS(SSptr);
        SSgroup.resize(J, SuffStat(SS.getprior()));
    }
    if (nrhs == 7)
    {
        for (int j = 0; j < J; ++j)
            SSgroup[j] = str2SS(mxGetCell(SSgroupptr, j));
    }
    
    // Call various versions of clustering algorithms depending on the arguments
    double *algptr = (double*)mxGetPr(prhs[1]);
    int algval = (int) algptr[0];
    mexstreambuf mexout;
    ostream mout(&mexout);

    try
    {
        if (algval == GMC)
        {
          if (nrhs == 2)
            F = learnGMC(X, qZ, SSgroup, SS, SPARSDEF, DIAGDEF, VERBDEF, mout);
          else if (nrhs == 3)
            F = learnGMC(X, qZ, SSgroup, SS, sparsptr[0], DIAGDEF, VERBDEF, 
                         mout);
          else if (nrhs == 4)
            F = learnGMC(X, qZ, SSgroup, SS, sparsptr[0], diagptr[0], VERBDEF, 
                         mout);
          else if (nrhs >= 5)
            F = learnGMC(X, qZ, SSgroup, SS, sparsptr[0], diagptr[0], 
                         verbptr[0], mout);
                         
        }
        else if (algval == SGMC)
        {
          if (nrhs == 2)
            F = learnSGMC(X, qZ, SSgroup, SS, SPARSDEF, DIAGDEF, VERBDEF, mout);
          else if (nrhs == 3)
            F = learnSGMC(X, qZ, SSgroup, SS, sparsptr[0], DIAGDEF, VERBDEF, 
                         mout);
          else if (nrhs == 4)
            F = learnSGMC(X, qZ, SSgroup, SS, sparsptr[0], diagptr[0], VERBDEF, 
                         mout);
          else if (nrhs >= 5)
            F = learnSGMC(X, qZ, SSgroup, SS, sparsptr[0], diagptr[0], 
                         verbptr[0], mout);
        }
        else
            throw logic_error("Wrong algorithm type specified!");
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
    mxArray *SSmx  = mxCreateCellMatrix(1, J);
    
    for (int j = 0; j < J; ++j)
    {       
        mxSetCell(qZmx, j, eig2mat(qZ[j])); // copy qZ
        mxSetCell(SSmx, j, SS2str(SSgroup[j]));  // copy wj
    } 
    
    plhs[1] = qZmx;
    plhs[2] = SSmx;
    
    // Copy SS to a matlab compatible structure
    plhs[3] = SS2str(SS);
}
