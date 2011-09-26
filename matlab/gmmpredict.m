function pX = gmmpredict (X, gmm)
% GMMPREDICT Probability of observations X occuring, according to the Gaussian 
%   mixture model. This is an interface for a C++ library that implements the 
%   various Gaussian mixture based models.
%
%   pX = gmmpredict (X, gmm)
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
% Returns:
%   - pX [Nx1] probability of each observation occuring using this GMM. This
%        uses the Variational posterior approximation to the GMM parameters.
%
% Notes:
%   - If X is data that has not been used to train the GMM, you will have to
%     make sure it has been scaled by the same process/constants as the training
%     data.
%   - This does not use the exact predicitive density of the models, but rather 
%     the finite GMM learned by the models. For all intents and purposes this is 
%     almost exactly the same.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     01/04/2011
%
% See also VDPCLUSTER, GMCCLUSTER, GMMCLASSIFY

    if size(gmm.mu,2) ~= size(X,2),
        error('Dimensionality of GMM and X is not the same!');
    end

    % Convert gmm struct to cell arrays
    gmm = convgmma2c(gmm);

    % Run the mex file
    pX = gmmpredict_mex(X, gmm);
    
end