function [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagcov, ...
                                                verbose, SSorprior, SSgroup)
% GMCCLUSTER Groups of Mixtures Clustering model for Gaussian mixture models.  
% This is an interface for a C++ library that implements the GMC. 
%
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagov)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagov, verbose)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagov, verbose, 
%                                       clustwidth)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagov, verbose, SS)
%   [qZ, SSgroup, SS, F] = gmccluster (X, Alg, sparse, diagov, verbose, SS, 
%                                       SSgroup)
%   
% Inputs:
%   - X {Jx[NjxD]} cell array of observation/feature matrices. Nj is the number 
%       of elements in each group, j, D is the number of dimensions. 
%   - Alg is the algorithm to use, valid options are:
%       * 'GDIR' Groups of Mixtures Clustering model with a Generalised
%                Dirichlet prior on the group weights.
%       * 'SDIR' Groups of Mixtures Clustering model with a Symmetric Dirichlet 
%                prior on the group weights.
%   - sparse true = use sparse version of GMC (faster, slightly less accurate), 
%            false = use original, dense, GMC (default).
%   - diagcov true = use diagonal covariance, false = full covariance. This is 
%             optional, false is default.
%   - verbose true = print verbose output, 0 = no output. This is optional, 
%             default is 0.
%   - clustwidth is the prior notion of the width of the clusters. This is 
%                optional, and the default value is 1e-5 which is mostly data-
%                driven.
%   - SS is a sufficient statistics structure for the model. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .D        the dimensionality of the suff. stats.
%       .priorval the value of the cluster hyperprior (cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%   - SSgroup {J X SS} cell array of group sufficient statistic structures 
%             (optional).
%
% Returns (all are optional):
%   - qZ {Jx[NjxK]} probability of each observation belonging to each cluster.
%        This is the Variational posterior approximation to p(Z|X).
%   - SSgroup {J X SS} cell array of group sufficient statistic structures.
%   - SS is a sufficient statistics structure for the model as above.
%   - F [scalar] final free energy. 
%
% Notes:
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
%   [1] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Clustering Groups 
%       of Related Visual Datasets," unpublished, 2011.
%
% See also GMMCLUSTER, SS2GMM

    % Check for a cell array
    if ~iscell(X), error('X must be a cell array!'); end
    
    % Check to see if X is double precision
    if isa( X{1}, 'double' ) == false
      error( 'X must be double precision' );
    end
    
    % Parse Alg argument
    switch lower(Alg)
        case 'gdir'
            algval = 2;
        case 'sdir',
            algval = 3;
        otherwise
            error('Unknown algorithm specified!');
    end
    
    % Run the suitable version of groupcluster_mex depending on the arguments
    switch nargin
        case 2,
            
            [F qZ SSgroup SS] = clustergroup_mex(X, algval);
            
        case 3,
            
            [F qZ SSgroup SS] = clustergroup_mex(X, algval, logical(sparse));

        case 4,
            [F qZ SSgroup SS] = clustergroup_mex(X, algval, logical(sparse), ...
                                    logical(diagcov));            
        case 5,
            
            [F qZ SSgroup SS] = clustergroup_mex(X, algval, logical(sparse), ...
                                    logical(diagcov), logical(verbose));
                                
        case 6,
        
            if ~isstruct(SSorprior)
                SS.priorval = SSorprior;
                SS.K = 0;
                SS.D = size(X{1}, 2);
                SS.F = 0;
                SS.N_k = {};
                SS.ss1 = {};
                SS.ss2 = {};
            else
                SS = SSorprior;
            end

            [F qZ SSgroup SS] = clustergroup_mex(X, algval, logical(sparse), ...
                          logical(diagcov), logical(verbose), SS);         
                                
        case 7,
        
            if ~iscell(SSgroup),
                error('SSgroup must be a cell array!');
            elseif any(size(X) ~= size(SSgroup)),
                error('X and SSgroup must have the same no. of cell elements');
            end
    
            [F qZ SSgroup SS] = clustergroup_mex(X, algval, logical(sparse), ...
                        logical(diagcov), logical(verbose), SSorprior, SSgroup);
        
        otherwise
            
            error('Invalid number of input arguments.');
    end
    
end
