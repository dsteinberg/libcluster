function [Z qZ] = gmmclassify (X, gmm)
% GMMCLASSIFY Classify observations X, according to a Gaussian mixture model
%   This is an interface for a C++ library that implements the various Gaussian 
%   mixture based models.
%
%   [Z, qZ] = gmmclassify (X, gmm)
%   
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the 
%       number of dimensions. 
%   - gmm is the Gaussian Mixture Model structure. It has fields:
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster.
%       .mu     [KxD] cluster means.
%       .sigma  [DxDxK] cluster covariances.
%
% Returns (all are optional):
%   - Z [Nx1] labels. These are the most likely clusters for each observation
%       in X.
%   - qZ [NxK] probability of each observation beloning to each cluster. This is
%        the Variational posterior approximation to p(Z|X).
%
% Notes:
%   - If X is data that has not been used to train the model, you will have to
%     make sure it has been scaled by the same process/constants as the training
%     data.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     18/03/2011
%
% See also VDPCLUSTER, GMCCLUSTER, GMMPREDICT

    if size(gmm.mu,2) ~= size(X,2),
        error('Dimensionality of GMM and X is not the same!');
    end

    % Convert gmm struct to cell arrays
    gmm = convgmma2c(gmm);

    % Run the mex file
    qZ = gmmclassify_mex(X, gmm);
    [tmp, Z] = max(qZ,[],2);
    
end