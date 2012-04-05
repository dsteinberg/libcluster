/*! \brief Variational Dirichlet Process cluster Labels for the ACFR AUV
 *         pipeline.
 *
 *  This file calls the Variational Dirichlet Process (VDP) clustering algorithm
 *  using image features data file, "image_features.data" for the AUV post
 *  processing pipeline.
 *
 *  vdp_labels.cpp has been adapted from a file created by MVJ which was a
 *  placeholder, but did most of the file ops present in this file.
 *
 *  This program outputs two files:
 *   image_labels.data  --   The labels for each point of data, 0 is a no-label
 *   SS.data            --   The sufficient statistics learned by the VDP.
 *
 *
 * \note The SS.data output uses a subset of the dimensions, which have been
 *       transformed and standardised.
 *
 * \todo Find a nice way to document the transforms and standardisation
 *       coefficients used to create the labels and SS.
 *
 * \author Daniel Steinberg
 *         Australian Centre for Field Robotics
 *         The University of Sydney
 *
 * \date   13/08/2011
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

#include "seabed_slam_file_io.hpp"
#include "adt_file_utils.hpp"
#include "auv_config_file.hpp"
#include "libcluster.h"
#include "probutils.h"

using namespace std;
using namespace libplankton;
using namespace auv_data_tools;
using namespace Eigen;
using namespace libcluster;
using namespace probutils;

//
// Typedefs and Constants
//

const int FEATDIMS = 7 + (2*Image_Feats::LAB_LENGTH) + Image_Feats::LBP_LENGTH;
const int NTHREADS = 1;
const double CLUSTWIDTH = 0.01;
const double SCALEFACTOR = 10;   // scaling for features.

//
// Helper functions
//

static bool parse_args (
    int argc,
    char *argv[ ],
    string& imfeat_fname,
    string& config_fname
    )
{
  if (argc < 3)
  {
    cerr << "Too few parameters." << endl;
    return false;
  }

  for (int i = 1; i < argc; ++i)
  {
    if (i == 1)
      imfeat_fname = argv[i];
    else if (i == 2)
      config_fname = argv[i];
    else
    {
      cerr << "Error - unknown parameter: " << argv[i] << endl;
      return false;
    }
  }

  return true;
}


static void print_usage ()
{
  cout << "USAGE: vdp_labels <image_features.data> <vdp_labels.cfg>" << endl;
  cout << endl;
}


//
// Main
//

int main (int argc, char *argv[])
{

  // Parse command line arguments
  string imfeat_fname, config_fname;
  if (parse_args(argc, argv, imfeat_fname, config_fname) == false)
  {
    print_usage();
    exit(1);
  }

  // Read config file for prior cluster width setting
  Config_File cfgfile(config_fname);
  int nthreads = cfgfile.get_int("NUMBER_OF_THREADS", NTHREADS);
  double clustwidth = cfgfile.get_double("PRIOR_CLUSTER_WIDTH", CLUSTWIDTH);

  // Read in the image features
  vector<Image_Feats> imfeats;
  try
    { imfeats = read_image_feature_file(imfeat_fname); }
  catch(Seabed_SLAM_IO_Exception &e)
  {
    cerr << "ERROR - " << e.what() << endl;
    exit(1);
  }

  // Parsing in features, and transforming them
  cout << "Parsing features ... ";
  ArrayXb isvdata(imfeats.size());
  MatrixXd Xparse(imfeats.size(), FEATDIMS);
  for (unsigned int i=0; i < imfeats.size(); i++)
  {
    // Morphological features
    Xparse(i,0) = log(imfeats[i].sp_rgsty-1+numeric_limits<double>::epsilon());
    Xparse(i,1) = log(imfeats[i].m5_rgsty-1+numeric_limits<double>::epsilon());
    Xparse(i,2) = log(imfeats[i].m5_slope+numeric_limits<double>::epsilon());
    Xparse(i,3) = log(imfeats[i].m10_rgsty-1+numeric_limits<double>::epsilon());
    Xparse(i,4) = log(imfeats[i].m20_slope+numeric_limits<double>::epsilon());

    // LAB Features
    Xparse(i,5) = imfeats[i].stdgray;
    Xparse(i,6) = log(imfeats[i].segsize);
    Xparse.block(i,7,1,Image_Feats::LAB_LENGTH)
        = Map<RowVectorXd>(imfeats[i].meanmod, 1, Image_Feats::LAB_LENGTH);
    Xparse.block(i,7+Image_Feats::LAB_LENGTH,1,Image_Feats::LAB_LENGTH)
        = Map<RowVectorXd>(imfeats[i].stdmod, 1, Image_Feats::LAB_LENGTH);

    // LBP features
    Xparse.block(i,7+2*Image_Feats::LAB_LENGTH,1,Image_Feats::LBP_LENGTH)
        = Map<RowVectorXd>(imfeats[i].lbp, 1, Image_Feats::LBP_LENGTH);

    // Test for NaNs
    isvdata(i) = !isnan(Xparse.row(i).sum());
  }

  int nvalid = isvdata.count();
  cout << nvalid << '/' << imfeats.size() << " features kept." << endl << endl;

  // Remove NaNs
  MatrixXd X(nvalid, FEATDIMS);
  for (unsigned int i=0, vidx=0; i < imfeats.size(); ++i)
  {
    if (isvdata(i) == true)
    {
      X.row(vidx) = Xparse.row(i);
      ++vidx;
    }
  }

  // Standardise features
  RowVectorXd meanX  = mean(X);
  RowVectorXd stdevX = stdev(X);
  for (int i=0; i < X.rows(); ++i)
    X.row(i) = SCALEFACTOR * (X.row(i) - meanX).array() / stdevX.array();

  // Cluster features
  SuffStat SS(clustwidth);
  MatrixXd qZ;
  try
  {  
    //clock_t start = clock();
    learnVDP(X, qZ, SS, true, nthreads);
    //double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
    //cout << "VDP Elapsed time = " << stop << " sec." << endl;
  }
  catch (runtime_error e)
  {
    cerr << "Runtime error: " << e.what() << endl;
    exit(1);
  }
  catch (invalid_argument e)
  {
    cerr << "Invalid argument: " << e.what() << endl;
    exit(1);
  }
  cout << endl;

  // Create image label data
  vector<Image_Label> image_label_data;
  for (unsigned int i=0, vidx=0; i < imfeats.size(); ++i)
  {
    Image_Label label;

    label.pose_id          = imfeats[i].pose_id;
    label.pose_time        = imfeats[i].pose_time;
    label.left_image_name  = imfeats[i].left_image_name;
    label.right_image_name = imfeats[i].right_image_name;

    if (isvdata(i) == true)
    {
      qZ.row(vidx).maxCoeff(&label.label);
      ++label.label; // account for C++ style begin array from 0th element
      ++vidx;
    }
    else
      label.label = 0;

    image_label_data.push_back(label);
  }

  // Write the image label file and Sufficient Statistic file.
  try
  {
    // Image labels
    write_image_label_file("image_label.data", "", image_label_data);

    // SuffStat object
    ofstream SSfile("SS.data");
    SSfile << endl << "% Cluster Sufficient Statistics:" << endl; 
    SSfile << SS << endl;
    SSfile.close();
  }
  catch (Seabed_SLAM_IO_Exception &e)
    { cerr << "ERROR - " << e.what() << endl; }
}
