//
// Includes and symbolic constants
//

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

streamsize mexstreambuf::xsputn (const char *s, streamsize n)
{
	mexPrintf("%.*s", n, s);
	return n;
}


int mexstreambuf::overflow (int c)
{
	if (c != EOF)
    	mexPrintf("%.1s", &c);
	return 1;
}


int mexstreambuf::sync ()
{
    if (pbase() == pptr()) // If the buffers is syncronised, do nothing
    {
    	mexEvalString("drawnow;");  // print to the console now
        return 1;
    }
    else // Else try and write the contents of the buffer
    {
        int cnwrite = mexPrintf(pbase());
        mexEvalString("drawnow;");  // print to the console now
        return cnwrite ? 0 : 1;
    }
}


mxArray* eig2mat(const MatrixXd& X)
{
    mxArray *rmat = mxCreateDoubleMatrix(X.rows(), X.cols(), mxREAL);
    double *rmatptr = mxGetPr(rmat);

    // Do a lame as element-wise copy -- is there a better way?
    // Column Major copy
    for (int j=0; j<X.cols(); ++j)
        for (int i=0, coloff=(j*X.rows()); i<X.rows(); ++i)
            rmatptr[coloff+i] = X(i,j);

    return rmat;
}


mxArray* SS2str (const SuffStat& SS)
{
    const char** fnames = new const char*[6];
    fnames[0] = "K";
    fnames[1] = "priorval";
    fnames[2] = "F";
    fnames[3] = "N_k";
    fnames[4] = "ss1";
    fnames[5] = "ss2";

    mxArray* SSstr = mxCreateStructMatrix(1, 1, 6, fnames);
    delete[] fnames;

    unsigned int K = SS.getK();

    // Copy SS suff. stat. vectors to cell arrays
    mxArray *N_k = mxCreateCellMatrix(1, K);
    mxArray *ss1 = mxCreateCellMatrix(1, K);
    mxArray *ss2 = mxCreateCellMatrix(1, K);

    for (unsigned int k = 0; k < K; ++k)
    {
        mxSetCell(N_k, k, mxCreateDoubleScalar(SS.getN_k(k)));
        mxSetCell(ss1, k, eig2mat(SS.getSS1(k)));
        mxSetCell(ss2, k, eig2mat(SS.getSS2(k)));
    }

    // Copy these cell arrays to a structure. We do this because I don't want to
    //  touch ND arrays with this mex library! I'll leave it for matlab code...
    mxSetFieldByNumber(SSstr, 0, 0, mxCreateDoubleScalar(K));
    mxSetFieldByNumber(SSstr, 0, 1, mxCreateDoubleScalar(SS.getprior()));
    mxSetFieldByNumber(SSstr, 0, 2, mxCreateDoubleScalar(SS.getF()));
    mxSetFieldByNumber(SSstr, 0, 3, N_k);
    mxSetFieldByNumber(SSstr, 0, 4, ss1);
    mxSetFieldByNumber(SSstr, 0, 5, ss2);

    return SSstr;
}


SuffStat str2SS (const mxArray* SS)
{
    // Get D, K, the hyper-prior and F
    unsigned int K  = (unsigned int) *mxGetPr(mxGetField(SS, 0, "K"));
    double priorval = (double) *mxGetPr(mxGetField(SS, 0, "priorval"));
    double F        = (double) *mxGetPr(mxGetField(SS, 0, "F"));

    // Initialise the SS object
    SuffStat SSr(priorval);
    
    // Populate SS object if required
    if (K > 0)
    {
        // Get dimensions of suff. stats.
        const int* ss1D = mxGetDimensions(mxGetCell(
                            mxGetField(SS, 0, "ss1"), 0));
        const int* ss2D = mxGetDimensions(mxGetCell(
                            mxGetField(SS, 0, "ss2"), 0));

        for (unsigned int k = 0; k < K; ++k)
        {
            double N_k = (double) *mxGetPr(mxGetCell(
                            mxGetField(SS, 0, "N_k"), k));
            MatrixXd ss1 = Map<MatrixXd>(mxGetPr(mxGetCell(
                            mxGetField(SS, 0, "ss1"), k)), ss1D[0], ss1D[1]);
            MatrixXd ss2 = Map<MatrixXd>(mxGetPr(mxGetCell(
                            mxGetField(SS, 0, "ss2"), k)), ss2D[0], ss2D[1]);

            SSr.setSS(k, N_k, ss1, ss2);
            SSr.setF(F);
        }
    }

    return SSr;
}
