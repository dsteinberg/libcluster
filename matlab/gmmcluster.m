function [Z, qZ, SS, F] = gmmcluster (X, Alg, diagcov, verbose, clustwidth)
% GMMCLUSTER Bayesian Gaussian Mixture models (GMM). 
%   This is an interface for a C++ library that implements the Bayesian GMM [1],
%   and also the Variation Dirichlet Process for GMMs [2].
%
%   [Z, qZ, SS, F] = gmmcluster (X, Alg)
%   [Z, qZ, SS, F] = gmmcluster (X, Alg, diagcov)
%   [Z, qZ, SS, F] = gmmcluster (X, Alg, diagcov, verbose)
%   [Z, qZ, SS, F] = gmmcluster (X, Alg, diagcov, verbose, clustwidth)
%   
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the 
%       number of dimensions. 
%   - Alg is the algorithm to use, valid options are:
%       * 'GMM' which is the Bayesian Gaussian Mixture Model
%       * 'VDP' which is the Variational Dirichlet Process for GMMs
%   - diagcov true = use diagonal covariance, false = full covariance. This is 
%             optional, false is default.
%   - verbose true = print verbose output, false = no output. This is optional, 
%             default is false.
%   - clustwidth is the prior notion of the width of the clusters. This is 
%                optional, and the default value is 1e-5 which is mostly data-
%                driven.
%
% Returns (all are optional):
%   - Z [Nx1] labels. These are the most likely clusters for each observation
%       in X.
%   - qZ [NxK] probability of each observation belonging to each cluster. This 
%        is the Variational posterior approximation to p(Z|X).
%   - SS is a sufficient statistics structure. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .priorval the value of the cluster hyperprior (e.g. cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%   - F [scalar] final free energy. 
%
% Notes:
%   - The difference between the VDP and GMM algorithms is that the prior
%     over the mixture weights is different. The VDP uses a Stick-Breaking
%     prior, while the GMM uses a symmetric Dirichlet prior.
%   - I find that standardising or whitening all of the dimensions of X before 
%     clustering improves results. 
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     13/12/2011
%
% References:
%   [1] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
%       Springer Science+Business Media, 2006.
%
%   [2] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational 
%       Dirichlet process mixtures, Advances in Neural Information 
%       Processing Systems, vol. 19, p. 761, 2007.
%
% See also GMCCLUSTER, SS2GMM

    % Check to see if X is double precision
    if isa( X, 'double' ) == false
      error( 'X must be double precision' );
    end
    
    % Parse Alg argument
    switch lower(Alg)
        case 'vdp'
            algval = 0;
        case 'gmm',
            algval = 1;
        otherwise
            error('Unknown algorithm specified!');
    end
            
    % Run the suitable version of cluster_mex depending on the arguments
    switch nargin
        case 2,
            [F qZ SS] = cluster_mex(X, algval);
        case 3,
            [F qZ SS] = cluster_mex(X, algval, logical(diagcov));
        case 4,
            [F qZ SS] = cluster_mex(X, algval, logical(diagcov), ...
                          logical(verbose));
        case 5,
            [F qZ SS] = cluster_mex(X, algval, logical(diagcov), ...
                          logical(verbose), clustwidth);
        otherwise
            error('Invalid number of input arguments.');
    end
    
    % Find most likely qZ and assign it to z 
    [~, Z] = max(qZ,[],2);
    
end
