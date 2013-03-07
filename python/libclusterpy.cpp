#include <Eigen/Dense>
#include "distributions.h"
#include "libclusterpy.h"

//
// Namespaces
//

using namespace std;
using namespace Eigen;
using namespace distributions;
using namespace libcluster;
using namespace boost::python;


//
// Private Functions
//

MatrixXd numpy2MatrixXd (PyObject* X)
{
  if (PyArray_Check(X) == false)
    throw invalid_argument("PyObject is not an array!");

  return Map<MatrixXd> ((double*) PyArray_DATA(X), PyArray_DIMS(X)[0],
                   PyArray_DIMS(X)[1]);
}


vMatrixXd getmean (const vector<GaussWish>& clusters)
{
  vMatrixXd means;

  for (size_t k=0; k < clusters.size(); ++k)
    means.push_back(clusters[k].getmean());

  return means;
}


vMatrixXd getcov (const vector<GaussWish>& clusters)
{
  vMatrixXd covs;

  for (size_t k=0; k < clusters.size(); ++k)
    covs.push_back(clusters[k].getcov());

  return covs;
}


//
//  Public Converters
//

PyObject* ArrayXd2numpy::convert (const ArrayXd& X)
{
  npy_intp arsize[] = {X.size()};
  ArrayXd* X_ = new ArrayXd(X); // Copy to persistent array
  PyObject* Xp = PyArray_SimpleNewFromData(1, arsize, NPY_DOUBLE, X_->data());

  if (Xp == NULL)
    throw runtime_error("Cannot convert Eigen array to Numpy array!");

  return Xp;
}


PyObject* MatrixXd2numpy::convert (const MatrixXd& X)
{
  npy_intp arsize[] = {X.rows(), X.cols()};
  MatrixXd* X_ = new MatrixXd(X); // Copy to persistent array
  PyObject* Xp = PyArray_SimpleNewFromData(2, arsize, NPY_DOUBLE, X_->data());

  if (Xp == NULL)
    throw runtime_error("Cannot convert Eigen array to Numpy array!");

  return Xp;
}


PyObject* vMatrixXd2numpy::convert (const vMatrixXd& X)
{
  list* Xp = new list();

  for (size_t i = 0; i < X.size(); ++i)
    Xp->append(X[i]);

  return Xp->ptr();
}


//
//  Public Wrappers
//

tuple wrapperVDP (
    PyObject* X,
    const float clusterprior,
    const bool verbose,
    const int nthreads
    )
{
  // Convert X
  const MatrixXd X_ = numpy2MatrixXd(X);

  // Pre-allocate some stuff
  MatrixXd qZ;
  StickBreak weights;
  vector<GaussWish> clusters;

  // Do the clustering
  double f = learnVDP(X_, qZ, weights, clusters, clusterprior, verbose,
                      nthreads);

  // Return relevant objects
  return make_tuple(f, qZ, ArrayXd(weights.Elogweight().exp()),
                    getmean(clusters), getcov(clusters));
}
