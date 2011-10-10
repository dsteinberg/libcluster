// Interfaces for commonly used functions/classes between the various mex files.

#ifndef INTFCTNS_H
#define INTFCTNS_H

#include <Eigen/Dense>
#include <iostream>
#include "mex.h"
#include "libcluster.h"

// Globals and Symbolics

#define SPARSDEF    0
#define VERBDEF     0
#define CWIDTHDEF   0.01f

enum algs { VDP, BGMM, GMC, SGMC, INCGMC };

// Mex stream buffer class, prints using mexPrintf()
class mexstreambuf : public std::streambuf
{
public:
protected:
	virtual std::streamsize xsputn (const char* s, std::streamsize n);
	virtual int overflow (int c = EOF);
    virtual int sync ();
};


// Convert 2D double Eigen matrix to 2D double mxArray
mxArray* eig2mat(const Eigen::MatrixXd& X);


// GMM object to matlab struct
//  The matlab struct will be of the form:
//      gmm.K     = scalar double number of clusters
//      gmm.w     = {1xK} array of scalar weights
//      gmm.mu    = {1x[1xD]} array of means
//      gmm.sigma = {1x[DxD]} array of covariances
mxArray* gmm2str (const libcluster::GMM& gmm);


// Matlab struct to GMM object
//  The matlab struct will be of the form:
//      gmm.K     = scalar double number of clusters
//      gmm.w     = {1xK} array of scalar weights
//      gmm.mu    = {1x[1xD]} array of means
//      gmm.sigma = {1x[DxD]} array of covariances
libcluster::GMM str2gmm (const mxArray* gmm);


// I-GMC object to matlab struct TODO DOC
mxArray* igmc2str (const libcluster::IGMC& igmc);


// Matlab struct to I-GMC object TODO DOC
libcluster::IGMC str2igmc (const mxArray* igmc);

#endif // INTFCTNS_H
