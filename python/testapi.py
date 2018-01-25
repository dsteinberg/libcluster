#! /usr/bin/env python

# libcluster -- A collection of hierarchical Bayesian clustering algorithms.
# Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
#
# This file is part of libcluster.
#
# libcluster is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# libcluster is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with libcluster. If not, see <http://www.gnu.org/licenses/>.

""" Script to make sure libcluster runs properly using the python API.

    Author: Daniel Steinberg
    Date:   13/10/2013

"""

import numpy as np
import libclusterpy as lc


# Top level cluster parameters -- Globals.... whatev...
means = np.array([[0, 0], [5, 5], [-5, -5]])
sigma = [np.eye(2)] * 3
beta = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3],
                 [1.0 / 2, 1.0 / 4, 1.0 / 4],
                 [1.0 / 4, 1.0 / 4, 1.0 / 2]])


def testmixtures():
    """ The test function. """

    print("Testing mixtures ------------------\n")

    # Create points from clusters
    W = gengmm(10000)

    # Test VDP
    print("------------ Test VDP -------------")
    f, qZ, w, mu, cov = lc.learnVDP(W, verbose=True)
    print("")
    printgmm(w, mu, cov)

    # Test BGMM
    print("------------ Test BGMM ------------")
    f, qZ, w, mu, cov = lc.learnBGMM(W, verbose=True)
    print("")
    printgmm(w, mu, cov)


def testgroupmix():

    print("Testing group mixtures ------------\n")

    # Create points from clusters
    J = 4   # Groups
    W = [gengmm(2000) for j in range(J)]

    # Test GMC
    print("------------ Test GMC -------------")
    f, qZ, w, mu, cov = lc.learnGMC(W, verbose=True)
    print("")
    printgmm(w, mu, cov)

    # Test SGMC
    print("------------ Test SGMC ------------")
    f, qZ, w, mu, cov = lc.learnSGMC(W, verbose=True)
    print("")
    printgmm(w, mu, cov)


def testmultmix():
    """ The the models that cluster at multiple levels. Just using J=1. """

    # Generate top-level clusters
    I = 200
    Ni = 100
    betas, Y = gensetweights(I)

    # Create points from clusters
    W = np.zeros((I, means.shape[1]))
    X = []
    for i in range(I):
        W[i, :] = np.random.multivariate_normal(means[Y[i]], sigma[Y[i]], 1)
        X.append(gengmm(Ni, betas[i, :]))

    # Test SCM
    print("------------ Test SCM -------------")
    f, qY, qZ, wi, ws, mu, cov = lc.learnSCM([X], trunc=30, verbose=True)
    print("")
    printgmm(ws, mu, cov)

    # Test MCM
    print("------------ Test MCM -------------")
    f, qY, qZ, wi, ws, mui, mus, covi, covs = lc.learnMCM([W], [X], trunc=30,
                                                          verbose=True)
    print("\nTop level mixtures:")
    printgmm(wi, mui, covi)
    print("Bottom level mixtures:")
    printgmm(ws, mus, covs)


def gengmm(N, weights=None):
    """ Make a random GMM with N observations. """

    K = len(sigma)
    pi = np.random.rand(K) if weights is None else weights
    pi /= pi.sum()
    Nk = np.round(pi * N)
    Nk[-1] = N - Nk[0:-1].sum()

    X = [np.random.multivariate_normal(means[k, :], sigma[k], int(Nk[k]))
         for k in range(K)]

    return np.concatenate(X)


def gensetweights(I):
    """ Generate sets of similar weights. """

    T = beta.shape[0]
    pi = np.random.rand(T)
    pi /= pi.sum()
    Nt = np.round(pi * I)
    Nt[-1] = I - Nt[0:-1].sum()

    betas = []
    Y = []
    for t in range(T):
        Y += int(Nt[t]) * [t]
        betas.append(int(Nt[t]) * [beta[t, :]])

    return np.concatenate(betas), Y


def printgmm(W, Mu, Cov):
    """ Print the parameters of a GMM. """

    Wnp = np.array(W)

    for i, (mu, cov) in enumerate(zip(Mu, Cov)):

        print("Mixture {0}:".format(i))
        if Wnp.ndim == 2:
            print(" weight --\n{0}".format(Wnp[i, :]))
        elif Wnp.ndim == 3:
            print(" group weights --\n{0}".format(Wnp[:, i, :]))
        print(" mean --\n{0}\n cov --\n{1}\n".format(mu, cov))


if __name__ == "__main__":
    testmixtures()
    testgroupmix()
    testmultmix()
