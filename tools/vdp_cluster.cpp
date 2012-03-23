// TODO:
//  - create more of an interactive command line interface, including a
//     help switch.

#include <fstream>
#include <time.h>
#include <cstring>
#include <wordexp.h>
#include "libcluster.h"


//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace libcluster;


//
// Function Prototypes
//

// Parse the features file
MatrixXd parse_features (const char* fname);


// Remove an tildes in paths, and replace with users home directory
char* replace_tilde (char* pathstr);


//
// Functions
//

// Main
int main (int argc, char* argv[])
{
  // Parse arguments
  char *ffile, *rfile; // input and output files
  if (argc == 3)
  {
    ffile = argv[1];
    rfile = argv[2];
  }
  else
    throw logic_error("Invalid number of arguments.");

  // Handle tilde ('~') in paths
  ffile = replace_tilde(ffile);
  rfile = replace_tilde(rfile);

  // Read in data file
  cout << "Reading feature file..." << flush;
  MatrixXd X = parse_features(ffile);
  cout << "done. X is [ " << X.rows() << " x " << X.cols() << " ]." << endl;

  // Cluster observations using VDP
  SuffStat SS;
  MatrixXd qZ;
  double F;
  clock_t start = clock();

  try
    { F = learnVDP(X, qZ, SS, false, true); }
  catch (logic_error e)
    { throw; }
  catch (runtime_error e)
    { throw; }

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Elapsed time = " << stop << " sec." << endl;

  // Save result data
  ofstream resfile(rfile);
  resfile << "N = " << qZ.rows() << endl; // Number of observations
  resfile << "K = " << qZ.cols() << endl; // Number of clusters
  resfile << "F = " << F << endl;         // Final free energy
  resfile << "p(z|x) = " << endl;         // Probability of class label
  resfile << qZ << endl;
  resfile.close();

  return 0;
}


MatrixXd parse_features (const char* fname)
{

  ifstream datfile(fname);

  if (datfile.is_open() == false)
    throw invalid_argument(string("Cannot open file: ").append(fname));

  // Get number of observations
  char c1, c2;
  int N, D;

  // Get number of observations
  datfile >> c1 >> c2;
  if ((c1 == 'N') & (c2 == '='))
    datfile >> N;
  else
    throw invalid_argument("Incorrect file format!");
  datfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get remainder of line

  // Get number of dimensionality
  datfile >> c1 >> c2;
  if ((c1 == 'D') & (c2 == '='))
    datfile >> D;
  else
    throw invalid_argument("Incorrect file format!");
  datfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // get remainder of line

  // Read in file element by element, parse file for consistency too.
  MatrixXd X(N,D);
  int d = 0, n = 0;
  string line;
  while (getline(datfile,line))
  {
    stringstream row(line);
    while (row.good())
    {
      row >> X(n,d);
      ++d;
    }

    // test for correct number of dimensions
    if ((d != D) & (d != 0))
      throw invalid_argument("Incorrect number of dimensions in data!");

    d=0;
    ++n;
    if (n >= N)
      break;
  }

  // test for correct number of rows
  if (n < (N-1))
    throw invalid_argument("Incorrect number of rows in data!");

  // Close the files
  datfile.close();

  return X;
}


char* replace_tilde (char* pathstr)
{
  char* newpath;
  if (pathstr[0] == '~')
  {
    wordexp_t file_path;
    wordexp(pathstr, &file_path, 0);
    newpath = file_path.we_wordv[0];
  }
  else
    newpath = pathstr;

  return newpath;
}
