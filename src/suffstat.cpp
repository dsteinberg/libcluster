#include <boost/math/special_functions.hpp>
#include "libcluster.h"
#include "probutils.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace boost::math;
using namespace probutils;


//
// Public Functions
//

ostream& libcluster::operator<< (ostream& s, const libcluster::SuffStat& SS)
{
  // K and D of the SS
  s << "K = " << SS.getK() << endl;

  // Numbers of each cluster
  s << "N_k = [ ";
  for (unsigned int k = 0; k < SS.getK(); ++k)
    s << SS.getN_k(k) << ' ';
  s << " ]" << endl << endl;

  // Sufficient statistic 1 of each cluster
  for (unsigned int k=0; k < SS.getK(); ++k)
    s << "Suff. Stat. 1(" << (k+1) << ") = " << endl << SS.getSS1(k) << endl;

  // Sufficient statistic 1 of each cluster
  for (unsigned int k=0; k < SS.getK(); ++k)
    s << "Suff. Stat. 2(" << (k+1) << ") = " << endl << SS.getSS2(k) << endl;

  return s;
}


//
// Public Member Functions
//

libcluster::SuffStat::SuffStat (double prior)
  : K(0), F(0), priorval(prior) {}


void libcluster::SuffStat::setSS (
    unsigned int k,
    double N,
    const MatrixXd& suffstat1,
    const MatrixXd& suffstat2
    )
{
  // Consistency checking
  if ( (this->K > 0) &&
         ((suffstat1.rows() != this->SS1[0].rows())
       || (suffstat1.cols() != this->SS1[0].cols())
       || (suffstat2.rows() != this->SS2[0].rows())
       || (suffstat2.cols() != this->SS2[0].cols())) )
    throw invalid_argument("Inconsistent dimensionalities!");

  // Resize if necessary
  if (k >= this->K)
  {
    this->N_k.resize(k+1, 0);
    this->SS1.resize(k+1, MatrixXd::Zero(suffstat1.rows(), suffstat1.cols()));
    this->SS2.resize(k+1, MatrixXd::Zero(suffstat2.rows(), suffstat2.cols()));
    this->K = k+1;
  }

  // Copy values
  this->N_k[k] = N;
  this->SS1[k] = suffstat1;
  this->SS2[k] = suffstat2;
}


double libcluster::SuffStat::getN_k (unsigned int k) const
{
  if (k > this->K)
    throw invalid_argument("Index k is greater than number of suff. stats.");

  return this->N_k[k];
}


const MatrixXd& libcluster::SuffStat::getSS1 (unsigned int k) const
{
  if (k > this->K)
    throw invalid_argument("Index k is greater than number of suff. stats.");

  return this->SS1[k];
}


const MatrixXd& libcluster::SuffStat::getSS2 (unsigned int k) const
{
  if (k > this->K)
    throw invalid_argument("Index k is greater than number of suff. stats.");

  return this->SS2[k];
}


void libcluster::SuffStat::addSS (const libcluster::SuffStat& SS)
{
  if (SS.K == 0)
    return;
  if ( (this->K > 0) && (this->compcheck(SS) == false) )
    throw invalid_argument("Suff. stats. are not compatible for addition!");

  for (unsigned int k=0; k < SS.K; ++k)
  {
    if (k < this->K)
    {
      this->N_k[k] += SS.N_k[k];
      this->SS1[k] += SS.SS1[k];
      this->SS2[k] += SS.SS2[k];
    }
    else
      this->setSS(k, SS.N_k[k], SS.SS1[k], SS.SS2[k]);
  }
}


void libcluster::SuffStat::subSS (const libcluster::SuffStat& SS)
{
  if (SS.K == 0)
      return; // Do nothing, SS has no information/uninstantiated.
  if (this->compcheck(SS) == false)
    throw invalid_argument("Suff. stats. are not compatible for subtraction!");

  for (unsigned int k=0; k < SS.K; ++k)
  {
    this->N_k[k] -= SS.N_k[k];
    this->SS1[k] -= SS.SS1[k];
    this->SS2[k] -= SS.SS2[k];
  }
}


void libcluster::SuffStat::addF(const SuffStat &SS)
{
  if (this->compcheck(SS) == false)
    throw invalid_argument("Suff. stats. are not compatible!");

  this->F += SS.F;
}


void libcluster::SuffStat::subF(const SuffStat &SS)
{
  if ((this->compcheck(SS) == false) || (this->K != SS.K))
    throw invalid_argument("Suff. stats. are not compatible!");

  this->F -= SS.F;
}


//
// Private Member Functions
//

bool libcluster::SuffStat::compcheck (const SuffStat &SS)
{
  return    (this->SS1[0].cols() == SS.SS1[0].cols())
         && (this->SS1[0].rows() == SS.SS1[0].rows())
         && (this->SS2[0].cols() == SS.SS2[0].cols())
         && (this->SS2[0].rows() == SS.SS2[0].rows())
         && (this->priorval == SS.priorval);
}
