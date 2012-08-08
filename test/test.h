#ifndef TEST_H
#define TEST_H

#include "libcluster.h"
#include "probutils.h"

//
// Common Test Functions
//

// Print a list of group weights to the command line from a SuffStat object
void printwj(const libcluster::vSuffStat& SSgroup)
{

  std::cout << std::endl;

  for (unsigned int j = 0; j < SSgroup.size(); ++j)
  {
    // Get number of Observations
    Eigen::RowVectorXd wj = Eigen::RowVectorXd::Zero(SSgroup[j].getK());

    for (unsigned int k = 0; k < SSgroup[j].getK(); ++k)
      wj(k) = SSgroup[j].getNk(k);

    std::cout << "w_group(" << j << ") = " << wj/wj.sum() << std::endl;
  }
}


// Print maximum likelihood GMM parameters from a SuffStat object
void printGMM (const libcluster::SuffStat& SS)
{
  // Get number of Dimensions and Observations
  bool isdiag = SS.getSS2(0).rows() == 1;
  double N = 0;
  for (unsigned int k = 0; k < SS.getK(); ++k)
    N += SS.getNk(k);

  // Print Mixture properties
  for (unsigned int k = 0; k < SS.getK(); ++k)
  {
    double w = SS.getNk(k)/N;
    Eigen::RowVectorXd mean = SS.getSS1(k)/SS.getNk(k);
    Eigen::MatrixXd cov;

    if (isdiag == true)
      cov = Eigen::MatrixXd(Eigen::RowVectorXd((SS.getSS2(k)/SS.getNk(k)
                              - mean.array().square().matrix())).asDiagonal());
    else
      cov = SS.getSS2(k)/SS.getNk(k) - mean.transpose()*mean;

    std::cout << std::endl << "w_k(" << k << ") = " << w << std::endl
         << "mu_k(" << k << ") = " << mean << std::endl
         << "sigma_k(" << k << ") = " << std::endl << cov << std::endl;
  }

  std::cout << std::endl;
}


#endif // TEST_H
