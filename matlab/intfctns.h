// Interfaces for commonly used functions/classes between the various mex files.

#ifndef INTFCTNS_H
#define INTFCTNS_H

#include <Eigen/Dense>
#include <iostream>
#include "mex.h"
#include "libcluster.h"


// Globals and Symbolics

enum single_algs { VDP = 0, BGMM = 1, DGMM = 2, BEMM = 3 };
enum group_algs  { GMC = 0, SGMC = 1, DGMC = 2, EGMC = 3 };
enum topic_algs  { TCM = 0 };

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


// SuffStat object to matlab struct
//  The matlab struct will be of the form:
//      SS.K        = scalar double number of clusters
//      SS.priorval = prior cluster hyperparameter
//      SS.N_k      = {1xK} array of observation counts
//      SS.x_k      = {1x[1xD]} array of observation sums
//      SS.xx_k     = {1x[DxD]} array of observation sum outer products
mxArray* SS2str (const libcluster::SuffStat& SS);


// Matlab struct to SuffStat object
//  The matlab struct will be of the form:
//      SS.K        = scalar double number of clusters
//      SS.priorval = prior cluster hyperparameter
//      SS.N_k      = {1xK} array of observation counts
//      SS.x_k      = {1x[1xD]} array of observation sums
//      SS.xx_k     = {1x[DxD]} array of observation sum outer products
libcluster::SuffStat str2SS (const mxArray* SS);

#endif // INTFCTNS_H
