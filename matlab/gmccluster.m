function [qZ, wj, gmm, F] = gmccluster (X, sparse, verbose, clustwidth)
% GMCCLUSTER Grouped Mixtures Clustering model for Gaussian mixture models. This 
% is an interface for a C++ library that implements the GMC. 
%
%   [qZ, wj, gmm, F] = gmccluster (X)
%   [qZ, wj, gmm, F] = gmccluster (X, sparse)
%   [qZ, wj, gmm, F] = gmccluster (X, sparse, verbose)
%   [qZ, wj, gmm, F] = gmccluster (X, sparse, verbose, clustwidth)
%   
% Inputs:
%   - X {Jx[NjxD]} cell array of observation/feature matrices. Nj is the number 
%       of elements in each group, j, D is the number of dimensions. 
%   - sparse 1 = use sparse version of GMC (faster, slightly less accurate), 
%            0 = use original, dense, GMC (default).
%   - verbose 1 = print verbose output, 0 = no output. This is optional, default
%             is 0.
%   - clustwidth is the prior notion of the width of the clusters relative to 
%                the principal eigen value of the data. This is optional, and 
%                the default value is 0.01. Typically a good range for this 
%                parameter is [0.01 1]. 
%
% Returns (all are optional):
%   - qZ {Jx[NjxK]} probability of each observation beloning to each cluster. 
%        This is the Variational posterior approximation to p(Z|X).
%   - wj {Jx[1xK]} weights of each cluster, k, in each group, j.
%   - gmm is the Gaussian Mixture Model structure. It has fields:
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster in the entire model.
%       .mu     [KxD] cluster means.
%       .sigma  [DxDxK] cluster covariances.
%   - F [scalar] final free energy. 
%
% Notes:
%   - I find that standardising all of the dimensions of X before clustering 
%     improves results dramatically. I.e., 
%
%     for j = 1:J,
%       X{j} = 10*(X{j} - repmat(mean_X,size(X{j},1),1))  ... 
%               ./ repmat(std_X,size(X{j},1),1);
%     end
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     19/04/2011
%
% References:
%   [1] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Hierarchal Bayesian
%       mixtures for clustering multiple related datasets." NIPS 2011
%       Submission, June 2011.
%
% See also SGMCCLUSTER, GMMCLASSIFY, GMMPREDICT

    if ~iscell(X), error('X must be a cell array!'); end

    if ~iscell(X), error('X must be a cell array!'); end
    
    % Run the suitable version of cmgcluster_mex depending on the arguments
    if nargin == 1,
        [F qZ wj gmm] = clustergroup_mex(X, 2);
    elseif nargin == 2,
        [F qZ wj gmm] = clustergroup_mex(X, 2, logical(sparse));
    elseif nargin == 3,
        [F qZ wj gmm] = clustergroup_mex(X, 2, logical(sparse), ...
                                logical(verbose));
    elseif nargin == 4,
        [F qZ wj gmm] = clustergroup_mex(X, 2, logical(sparse), ...
                               	logical(verbose), clustwidth);
    else
        error('Invalid number of input arguments.');
    end
    
    % Convert gmm structure
    if nargout > 3, gmm = convgmmc2a(gmm); end

end