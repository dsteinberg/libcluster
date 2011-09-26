function [Z, qZ, gmm, F] = vdpcluster (X, verbose, clustwidth)
% VDPCLUSTER Variational Dirichlet Process (VDP) for Gaussian mixture models. 
%   It is an interface for a C++ library that implements the VDP as specified 
%   by [1]. 
%
%   [Z, qZ, gmm, F] = vdpcluster (X)
%   [Z, qZ, gmm, F] = vdpcluster (X, verbose)
%   [Z, qZ, gmm, F] = vdpcluster (X, verbose, clustwidth)
%   
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the 
%       number of dimensions. 
%   - verbose 1 = print verbose output, 0 = no output. This is optional, default
%             is 0.
%   - clustwidth is the prior notion of the width of the clusters relative to 
%                the principal eigen value of the data. This is optional, and 
%                the default value is 0.01. Typically a good range for this 
%                parameter is [0.01 1]. 
%
% Returns (all are optional):
%   - Z [Nx1] labels. These are the most likely clusters for each observation
%       in X.
%   - qZ [NxK] probability of each observation beloning to each cluster. This is
%        the Variational posterior approximation to p(Z|X).
%   - gmm is the Gaussian Mixture Model structure. It has fields:
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster.
%       .mu     [KxD] cluster means.
%       .sigma  [DxDxK] cluster covariances.
%   - F [scalar] final free energy. 
%
% Notes:
%   - I find that standardising all of the dimensions of X before clustering 
%     improves results dramatically. I.e.,
%   
%       X = 10*(X - repmat(mean(X),size(X,1),1)) ./ repmat(std(X),size(X,1),1);
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     18/03/2011
%
% References:
%   [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational 
%       Dirichlet process mixtures, Advances in Neural Information 
%       Processing Systems, vol. 19, p. 761, 2007.
%
% See also GMCCLUSTER, GMMCLASSIFY, GMMPREDICT

    % Run the suitable version of vdpcluster_mex depending on the arguments
    if nargin == 1,
        [F qZ gmm] = vdpcluster_mex(X);
    elseif nargin == 2,
        [F qZ gmm] = vdpcluster_mex(X, logical(verbose));
    elseif nargin == 3,
        [F qZ gmm] = vdpcluster_mex(X, logical(verbose), clustwidth);
    else
        error('Invalid number of input arguments.');
    end
    
    % Find most likely qz and assign it to z 
    [tmp, Z] = max(qZ,[],2);
    
    % Convert gmm structure
    if nargout > 2, gmm = convgmmc2a(gmm); end

end