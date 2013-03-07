#ifndef LIBCLUSTERPY_H
#define LIBCLUSTERPY_H

#include <omp.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "libcluster.h"

//
//  Type converters
//

struct ArrayXd2numpy
{
  static PyObject* convert (const Eigen::ArrayXd& X);
};


struct MatrixXd2numpy
{
  static PyObject* convert (const Eigen::MatrixXd& X);
};


struct vMatrixXd2numpy
{
  static PyObject* convert (const libcluster::vMatrixXd& X);
};


//
//  Wrappers
//

boost::python::tuple wrapperVDP (
    PyObject* X,
    const float clusterprior = libcluster::PRIORVAL,
    const bool verbose = false,
    const int nthreads = omp_get_max_threads()
    );

BOOST_PYTHON_FUNCTION_OVERLOADS (wrapperVDPover, wrapperVDP, 1, 4)


//
//  Module definition
//

BOOST_PYTHON_MODULE (libclusterpy)
{
  import_array();

  // To-python converters
  boost::python::to_python_converter<Eigen::ArrayXd, ArrayXd2numpy>();
  boost::python::to_python_converter<Eigen::MatrixXd, MatrixXd2numpy>();
  boost::python::to_python_converter<libcluster::vMatrixXd, vMatrixXd2numpy>();

  // Functions
  boost::python::def("learnVDP", wrapperVDP, wrapperVDPover());

}

#endif
