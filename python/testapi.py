#! /usr/bin/env python
""" Script to make sure libcluster runs properly using the python API. 
    
    Author: Daniel Steinberg
    Date:   13/10/2013

"""

import numpy as np
import libclusterpy as lc


# Top level cluster parameters -- Globals.... whatev...
wmeans = np.array([[0, 0], [5, 5], [-5, -5]])
wsigma = [np.eye(2)] * 3
T = 3   # Clusters
Nt = 500  # number of points per cluster

def testmixtures():
    """ The test function. """

    print "Testing mixtures ------------------\n"

    # Create points from clusters
    W = [np.random.multivariate_normal(mean, cov, (Nt)) for mean, cov in 
            zip(wmeans, wsigma)]
    W = np.concatenate(W)

    # Test VDP 
    print "------------ Test VDP -------------"    
    f, qZ, w, mu, cov = lc.learnVDP(W, verbose=True)
    print ""
    printgmm(w, mu, cov)

    # Test BGMM
    print "------------ Test BGMM ------------"    
    f, qZ, w, mu, cov = lc.learnBGMM(W, verbose=True)
    print ""
    printgmm(w, mu, cov)


def testgroupmix():

    print "Testing group mixtures ------------\n"

    J = 4   # Groups


    # Create points from clusters
    W = [np.random.multivariate_normal(mean, cov, (Nt)) for mean, cov in 
            zip(wmeans, wsigma)]

    W = makegroups(W, J)

    # Test GMC
    print "------------ Test GMC -------------"    
    f, qZ, w, mu, cov = lc.learnGMC(W, verbose=True)
    print ""
    printgmm(w, mu, cov)

    # Test SGMC
    print "------------ Test SGMC ------------"    
    f, qZ, w, mu, cov = lc.learnSGMC(W, verbose=True)
    print ""
    printgmm(w, mu, cov)


def makegroups(X, J):
    """ Divide X into J random groups, X should be grouped by cluster. """
    
    # Get grou and cluster properties
    K = len(X)
    Nk = np.array([len(x) for x in X])
    N = Nk.sum()

    # Randomly assign observation-cluster counts to groups
    pi_j = np.random.rand(J, K)
    pi_j /= (pi_j.sum(axis=0)[:, None]).T
    Njk = np.round(pi_j * Nk)
    Njk[-1, :] = Nk - Njk[0:-1, :].sum(axis=0)

    # Now make the groups
    Xgroups = []
    Nk -= 1
    
    for j in xrange(0, J):
        Xj = []
        for k in xrange(0, K):
            while Njk[j, k] > 0:
                Xj.append(X[k][Nk[k]])
                Njk[j, k] -= 1
                Nk[k] -= 1
        Xgroups.append(np.array(Xj))

    return Xgroups


def printgmm(W, Mu, Cov):
    """ Print the parameters of a GMM. """

    Wnp = np.array(W)

    for i, (mu, cov) in enumerate(zip(Mu, Cov)):
    
        print "Mixture {0}:".format(i)
        if Wnp.ndim == 2:
            print " weight --\n{0}".format(Wnp[i, :])
        elif Wnp.ndim == 3:
            print " group weights --\n{0}".format(Wnp[:, i, :])
        print " mean --\n{0}\n cov --\n{1}\n".format(mu, cov)


if __name__ == "__main__":
    testmixtures()
    testgroupmix()
