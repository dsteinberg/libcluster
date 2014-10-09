/*
 * libcluster -- A collection of hierarchical Bayesian clustering algorithms.
 * Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
 *
 * This file is part of libcluster.
 *
 * libcluster is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * libcluster is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcluster. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TESTDATA_H
#define TESTDATA_H

#include <Eigen/Dense>
#include <vector>

// Populates some test data -- Twelve groups of 3 identity 2D covariance
//  Gaussians.
// TODO: MORE DESCRIPTION
void makeXdata (
    Eigen::MatrixXd& Xcat,            // [Group one; Group 2; ..] concatenated
    std::vector<Eigen::MatrixXd>& X   // {Group one, Group 2, ...} in a vector
    )
{

  X.clear();

  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[0]  <<  2.0243,    1.9085,
           -2.3595,    0.1222,
           -0.5100,    1.0470,
           -1.3216,   -0.2269,
           -0.6361,   -0.1625,
            0.3179,    0.6901,
            0.1380,    0.5558,
          -10.5718,   11.0533,
          -10.2500,    9.2511,
          -11.5693,    9.0637;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[1]  << -9.5793,   10.5411,
           -9.5993,    8.4591,
           -9.9049,    9.7969,
            9.4565,    8.7706,
            9.0881,    9.7290,
           10.6527,    9.1000,
            9.2657,    9.7143,
           10.5406,    9.5376,
           10.9758,    9.5902,
            9.8431,    9.4965;

  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[2]  << -0.7107,   -1.1203,
            0.7770,   -1.5327,
            0.6224,   -1.0979,
            0.6474,   -1.4158,
           -0.4256,    0.0596,
            1.0486,   -0.4113,
            0.6607,   -0.3680,
          -11.3380,   10.4980,
           -9.9697,   12.7891,
           -9.1469,   10.7276;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[3]  << -9.5033,    9.5000,
           -8.9178,   10.3830,
           -9.0296,   10.4120,
           10.2778,   11.2333,
           10.6395,   10.6103,
            9.9190,   10.0591,
           10.5409,    8.5331,
            8.7374,    8.3742,
           11.1104,    8.0352,
            9.0104,   12.6052;

  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[4]  <<  2.5088,   -1.3610,
            1.0635,    0.7796,
            1.1569,    0.4394,
            0.0530,   -0.0896,
           -1.2884,    1.0212,
           -0.3712,   -0.8740,
           -0.7578,    0.4147,
           -9.5957,    9.2269,
          -10.7006,   10.8366,
          -11.6305,    8.8717;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[5]  << -10.5686,   10.4055,
           -9.1900,    9.6362,
           -9.8268,    9.4007,
           10.6263,    9.5506,
            9.7133,    9.9157,
            9.8027,    8.0080,
           10.4056,   10.8412,
            8.5807,    9.5853,
            9.2706,   11.9122,
           11.1473,    9.6091;

  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[6]  << -0.5640,    0.3484,
            0.5551,    0.3493,
           -0.5568,   -0.7292,
           -0.8951,    0.3268,
           -0.4093,   -0.5149,
           -0.1609,   -0.8964,
            0.4093,   -1.2033,
           -9.5957,    9.2269,
          -10.7006,   10.8366,
          -11.6305,    8.8717;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[7]  << -10.5055,    9.4104,
           -11.1933,   10.8535,
            -9.3530,    8.1470,
             8.1712,   10.9724,
            11.3845,   10.2570,
             9.9373,    9.0258,
            10.4489,    8.8536,
             9.6367,   10.5476,
             8.9794,   11.5651,
             6.9270,    8.3067;

  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[8]  << -0.9526,    1.0378,
            0.3173,   -0.8459,
            0.0780,   -0.1729,
            1.3244,   -1.2087,
           -0.2132,   -0.2971,
           -0.1345,   -3.2320,
           -1.1714,   -1.0870,
           -8.5400,    8.5755,
           -7.9500,   10.7174,
           -9.8795,    9.2221;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[9] <<  -10.3536,    9.7927,
            -9.9536,   10.2704,
           -10.7929,    9.3472,
            10.5979,   10.4092,
             8.7187,    8.8576,
             7.7967,    9.3751,
             9.4288,    8.8313,
            10.2140,   10.3926,
            10.9424,   11.3018,
            10.0937,    9.4064;


  X.push_back(Eigen::MatrixXd(10,2)); // [0 0], [-10 10]
  X[10] << -1.3853,   -1.4264,
            0.3105,   -1.0145,
           -0.2495,   -0.2133,
            0.5037,   -0.3253,
           -0.8927,    1.9444,
          -10.4698,   10.9297,
           -9.1136,    8.3942,
          -11.3852,   10.6615,
          -10.4774,    8.7309,
          -11.9568,   12.1385;

  X.push_back(Eigen::MatrixXd(10,2)); // [-10 10], [10 10]
  X[11] << -11.5505,   10.4772,
            -9.8284,    9.9287,
           -10.0621,    9.0617,
            -8.8010,   10.1614,
            -9.1983,    9.7318,
             9.5901,    8.8777,
             9.2887,   10.3062,
            10.0614,    8.8277,
             8.1539,    9.0390,
             9.6017,    9.3463;

  Xcat.setZero(120,2);
  const int J = X.size();
  for (int j=0; j < J; ++j)
    Xcat.block(j*10, 0, 10, 2) = X[j];

}

// Populates some more test data -- Two groups of 1 identity 2D covariance
//  Gaussians.
// TODO: MORE DESCRIPTION
void makeOdata (
    Eigen::MatrixXd& Ocat,            // [Group one; Group 2] concatenated
    std::vector<Eigen::MatrixXd>& O   // {Group one, Group 2} in a vector
    )
{
  O.clear();

  O.push_back(Eigen::MatrixXd(6,2));
  O[0] <<  5.4889,   5.8884,
          -4.6748,  -4.6808,
           6.0347,   3.8529,
          -5.7549,  -4.6871,
           5.7269,   3.9311,
          -3.6297,  -5.8649;

  O.push_back(Eigen::MatrixXd(6,2));
  O[1] <<  4.6966,   4.1905,
          -6.7115,  -5.0301,
           5.2939,   2.0557,
          -5.1022,  -5.1649,
           4.2127,   6.4384,
          -5.2414,  -4.3723;

  Ocat.setZero(12,2);
  Ocat.block(0, 0, 6, 2) = O[0];
  Ocat.block(6, 0, 6, 2) = O[1];

}

#endif // TESTDATA_H
