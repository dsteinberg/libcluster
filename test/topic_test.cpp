#include "libcluster.h"
#include "probutils.h"
#include "testdata.h"

//
// Namespaces
//


using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;


//
// Functions
//


// TEMP: THIS IS COPIED FROM CLUSTER_TEST.CPP
void printwj(const vector<SuffStat>& SSgroup)
{

  cout << endl;

  for (unsigned int j = 0; j < SSgroup.size(); ++j)
  {
    // Get number of Observations
    RowVectorXd wj = RowVectorXd::Zero(SSgroup[j].getK());

    for (unsigned int k = 0; k < SSgroup[j].getK(); ++k)
      wj(k) = SSgroup[j].getN_k(k);

    cout << "w_group(" << j << ") = " << wj/wj.sum() << endl;
  }
}


// TEMP: THIS IS COPIED FROM CLUSTER_TEST.CPP
void printGMM (const SuffStat& SS)
{
  // Get number of Dimensions and Observations
  bool isdiag = SS.getSS2(0).rows() == 1;
  double N = 0;
  for (unsigned int k = 0; k < SS.getK(); ++k)
    N += SS.getN_k(k);

  // Print Mixture properties
  for (unsigned int k = 0; k < SS.getK(); ++k)
  {
    double w = SS.getN_k(k)/N;
    RowVectorXd mean = SS.getSS1(k)/SS.getN_k(k);
    MatrixXd cov;

    if (isdiag == true)
      cov = MatrixXd(RowVectorXd((SS.getSS2(k)/SS.getN_k(k)
                              - mean.array().square().matrix())).asDiagonal());
    else
      cov = SS.getSS2(k)/SS.getN_k(k) - mean.transpose()*mean;

    cout << endl << "w_k(" << k << ") = " << w << endl
         << "mu_k(" << k << ") = " << mean << endl
         << "sigma_k(" << k << ") = " << endl << cov << endl;
  }

  cout << endl;
}


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vector<MatrixXd> X;
  makedata(Xcat, X);

  const unsigned I = X.size();

  SuffStat SS;
  vector<SuffStat> SSdocs(I, SuffStat());
  MatrixXd qY, classparams;
  vector<MatrixXd> qZ;
  clock_t start = clock();

  learnTCM(X, qY, qZ, SSdocs, SS, classparams, 4, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Topic Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Class proportions:" << endl;
  cout << qY.colwise().sum()/I << endl << endl;

  cout << endl << "Class parameters:" << endl;
  cout << classparams << endl << endl;

  cout << "Document cluster proportions:";
  printwj(SSdocs);
  cout << endl << "Cluster parameters:";
  printGMM(SS);

  return 0;
}
