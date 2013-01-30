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


//
// Functions
//


// Main
int main()
{

  // Populate test data from testdata.h
  MatrixXd Xcat, Ocat;
  vMatrixXd X, W;
  vvMatrixXd Xv(2);
  makeXdata(Xcat, X);
  makeOdata(Ocat, W);

  // Divide up X into 2 meta datasets
  for (unsigned int j = 0; j < X.size(); ++j)
  {
    if (j < (X.size()/2))
      Xv[0].push_back(X[j]);
    else
      Xv[1].push_back(X[j]);
  }

  vector<GDirichlet> iweights;
  vector<Dirichlet>  sweights;
  vector<GaussWish>  sclusters;
  vector<GaussWish>  iclusters;
  vMatrixXd qY;
  vvMatrixXd qZ;
  clock_t start = clock();

  learnMCM(W, Xv, qY, qZ, iweights, sweights, iclusters, sclusters, 10,
           PRIORVAL, PRIORVAL, true);

  double stop = (double)((clock() - start))/CLOCKS_PER_SEC;
  cout << "Topic Elapsed time = " << stop << " sec." << endl;

  cout << endl << "Image cluster proportions:" << endl;
  for (vector<GDirichlet>::iterator j = iweights.begin(); j<iweights.end(); ++j)
    cout << j->Elogweight().exp().transpose() << endl;

  cout << endl << "Segment cluster proportions per image cluster:" << endl;
  for (vector<Dirichlet>::iterator t = sweights.begin(); t<sweights.end(); ++t)
    cout << t->Elogweight().exp().transpose() << endl;

  cout << endl << "Image cluster means:" << endl;
  for (vector<GaussWish>::iterator t=iclusters.begin(); t<iclusters.end(); ++t)
    cout << t->getmean() << endl;

  cout << endl << "Image cluster covariances:" << endl;
  for (vector<GaussWish>::iterator t=iclusters.begin(); t<iclusters.end(); ++t)
    cout << t->getcov() << endl << endl;

  cout << endl << "Segment cluster means:" << endl;
  for (vector<GaussWish>::iterator k=sclusters.begin(); k<sclusters.end(); ++k)
    cout << k->getmean() << endl;

  cout << endl << "Segment cluster covariances:" << endl;
  for (vector<GaussWish>::iterator k=sclusters.begin(); k<sclusters.end(); ++k)
    cout << k->getcov() << endl << endl;
}
