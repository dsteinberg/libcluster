/*
 * libcluster -- A collection of Bayesian clustering algorithms
 * Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "libcluster.h"
#include "distributions.h"
#include "testdata.h"


//
// Namespaces
//


using namespace std;
using namespace Eigen;
using namespace libcluster;
using namespace distributions;


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat;
  vMatrixXd X;
  makeXdata(Xcat, X);

  // GMC
  vector<GDirichlet> weights;
  vector<GaussWish>  clusters;
  vMatrixXd qZgroup;
  clock_t start = clock();
  learnGMC (X, qZgroup, weights, clusters, PRIORVAL, false, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "GMC Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Cluster Weights:" << endl;
  for (vector<GDirichlet>::iterator j = weights.begin(); j < weights.end(); ++j)
    cout << j->Elogweight().exp().transpose() << endl;

  cout << endl << "Cluster means:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getmean() << endl;

  cout << endl << "Cluster covariances:" << endl;
  for (vector<GaussWish>::iterator k=clusters.begin(); k < clusters.end(); ++k)
    cout << k->getcov() << endl << endl;

  return 0;
}
