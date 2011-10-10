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
// Classes
//

// Inherited I-GMC class explicitly for copying it to a matlab structure and not
//  messing up the C++ interface
class IGMCmx : public IGMC
{
public:

    // create this object, and by extension an I-GMC object from a matlab igmc
    //  structure
    IGMCmx (const mxArray* igmc)
    {
        this->J = (int) *mxGetPr(mxGetField(igmc, 0, "J"));
        this->D = (int) *mxGetPr(mxGetField(igmc, 0, "D"));
        this->K = (int) *mxGetPr(mxGetField(igmc, 0, "K"));
        this->kappa   = *mxGetPr(mxGetField(igmc, 0, "kappa"));
        this->tau0    = *mxGetPr(mxGetField(igmc, 0, "tau0"));
        this->cwidthp = *mxGetPr(mxGetField(igmc, 0, "cwidthp"));

        Map<RowVectorXd> cmeanp(mxGetPr(mxGetField(igmc, 0, "cmeanp")),
                mxGetM(mxGetField(igmc, 0, "cmeanp")),
                mxGetN(mxGetField(igmc, 0, "cmeanp")));
        this->cmeanp = cmeanp;

        // These are user set variables - so we better check them!
        if (this->kappa < 0)
           mexErrMsgTxt("kappa must be greater than 0!");
        if (this->tau0 < 1)
            mexErrMsgTxt("tau0 must be greater than 1!");
        if (this->J < 1)
            mexErrMsgTxt("J must be greater than or equal to 1!");
        if (this->D < 1)
            mexErrMsgTxt("D must be greater than or equal to 1!");
        if (this->cwidthp <= 0)
            mexErrMsgTxt("cwidthp must be greater than 0!");
        if (cmeanp.cols() != this->D)
            mexErrMsgTxt("cmeanp must have a length of D!");

        // These should be automatically set by matlab scripts, we'll assume ok
        this->tau      = (int)    *mxGetPr(mxGetField(igmc, 0, "tau"));
        this->lambda   = (double) *mxGetPr(mxGetField(igmc, 0, "lambda"));
        this->rho      = (double) *mxGetPr(mxGetField(igmc, 0, "rho"));
        this->Fw       = (double) *mxGetPr(mxGetField(igmc, 0, "Fw"));
        this->Fxz      = (double) *mxGetPr(mxGetField(igmc, 0, "Fxz"));

        for (int k = 0; k < this->K; ++k)
        {
            this->N_s.push_back(*mxGetPr(mxGetCell(
                mxGetField(igmc, 0, "N_s"), k)));
            this->x_s.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(igmc, 0, "x_s"), k)), 1, this->D));
            this->xx_s.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(igmc, 0, "xx_s"), k)), this->D, this->D));

            this->w.push_back(*mxGetPr(mxGetCell(mxGetField(igmc, 0, "w"), k)));
            this->mu.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(igmc, 0, "mu"), k)), 1, this->D));
            this->sigma.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(igmc, 0, "sigma"), k)), this->D, this->D));
        }
    }


    // create this object from an I-GMC object
    IGMCmx (const IGMC& igmc) : IGMC(igmc) {}


    // Return a matlab structure created from an I-GMC
    mxArray* igmcmat () const
    {
        // Add the base GMM fields
        mxArray* igmcr = gmm2str(*this);

        // Add the scalar fields
        mxAddField(igmcr, "D");
        mxSetField(igmcr, 0, "D", mxCreateDoubleScalar(this->D));
        mxAddField(igmcr, "J");
        mxSetField(igmcr, 0, "J", mxCreateDoubleScalar(this->J));
        mxAddField(igmcr, "kappa");
        mxSetField(igmcr, 0, "kappa", mxCreateDoubleScalar(this->kappa));
        mxAddField(igmcr, "tau0");
        mxSetField(igmcr, 0, "tau0", mxCreateDoubleScalar(this->tau0));
        mxAddField(igmcr, "cwidthp");
        mxSetField(igmcr, 0, "cwidthp", mxCreateDoubleScalar(this->cwidthp));
        mxAddField(igmcr, "tau");
        mxSetField(igmcr, 0, "tau", mxCreateDoubleScalar(this->tau));
        mxAddField(igmcr, "rho");
        mxSetField(igmcr, 0, "rho", mxCreateDoubleScalar(this->rho));
        mxAddField(igmcr, "lambda");
        mxSetField(igmcr, 0, "lambda", mxCreateDoubleScalar(this->lambda));
        mxAddField(igmcr, "Fw");
        mxSetField(igmcr, 0, "Fw", mxCreateDoubleScalar(this->Fw));
        mxAddField(igmcr, "Fxz");
        mxSetField(igmcr, 0, "Fxz", mxCreateDoubleScalar(this->Fxz));

        // Add cmeanp array
        mxAddField(igmcr, "cmeanp");
        mxSetField(igmcr, 0, "cmeanp", eig2mat(this->cmeanp));

        // Add cell array fields
        mxArray *Nkc = mxCreateCellMatrix(1, K);
        mxArray *Xkc = mxCreateCellMatrix(1, K);
        mxArray *Rkc = mxCreateCellMatrix(1, K);

        for (int k = 0; k < K; ++k)
        {
            mxSetCell(Nkc, k, mxCreateDoubleScalar(this->N_s[k]));
            mxSetCell(Xkc, k, eig2mat(this->x_s[k]));
            mxSetCell(Rkc, k, eig2mat(this->xx_s[k]));
        }

        // Copy these cell arrays to a structure.
        mxAddField(igmcr, "N_s");
        mxSetField(igmcr, 0, "N_s", Nkc);
        mxAddField(igmcr, "x_s");
        mxSetField(igmcr, 0, "x_s", Xkc);
        mxAddField(igmcr, "xx_s");
        mxSetField(igmcr, 0, "xx_s", Rkc);

        return igmcr;
    }
};


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


mxArray* gmm2str (const GMM& gmm)
{
    const char** fnames = new const char*[4];
    fnames[0] = "K";
    fnames[1] = "w";
    fnames[2] = "mu";
    fnames[3] = "sigma";

    mxArray* gmmstr = mxCreateStructMatrix(1, 1, 4, fnames);
    delete[] fnames;

    int K = gmm.getK();
    int D = gmm.getD();

    // Copy GMM weight, mu and sigma vectors to cell arrays
    mxArray *w     = mxCreateCellMatrix(1, K);
    mxArray *mu    = mxCreateCellMatrix(1, K);
    mxArray *sigma = mxCreateCellMatrix(1, K);

    for (int k = 0; k < K; ++k)
    {
        mxSetCell(w, k, mxCreateDoubleScalar(gmm.getw(k)));
        mxSetCell(mu, k, eig2mat(gmm.getmu(k)));
        mxSetCell(sigma, k, eig2mat(gmm.getsigma(k)));
    }

    // Copy these cell arrays to a structure. We do this because I don't want to
    //  touch ND arrays with this mex library! I'll leave it for matlab code...
    mxSetFieldByNumber(gmmstr, 0, 0, mxCreateDoubleScalar(K));
    mxSetFieldByNumber(gmmstr, 0, 1, w);
    mxSetFieldByNumber(gmmstr, 0, 2, mu);
    mxSetFieldByNumber(gmmstr, 0, 3, sigma);

    return gmmstr;
}


GMM str2gmm (const mxArray* gmm)
{
    // Get D and K
    int K = (int) *mxGetPr(mxGetField(gmm, 0, "K"));
    int D = (int) mxGetN(mxGetCell(mxGetField(gmm, 0, "mu"), 0));

    // Convert GMM cell arrays to the required GMM object vectors
    vector<double> w;
    vector<RowVectorXd> mu;
    vector<MatrixXd> sigma;

    for (int k=0; k < K; ++k)
    {
        w.push_back(*mxGetPr(mxGetCell(mxGetField(gmm, 0, "w"), k)));
        mu.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(gmm, 0, "mu"), k)), 1, D));
        sigma.push_back(Map<MatrixXd>(mxGetPr(
                mxGetCell(mxGetField(gmm, 0, "sigma"), k)), D, D));
    }

    // Initialise the GMM object
    GMM gmmr(mu, sigma, w);
    MatrixXd pZ;

    return gmmr;
}


mxArray* igmc2str (const IGMC& igmc)
{
    IGMCmx igmccp(igmc);
    return igmccp.igmcmat();
}


IGMC str2igmc (const mxArray* igmc) { return IGMCmx(igmc); }
