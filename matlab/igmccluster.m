function [IGMC, ischanged] = igmccluster (X, IGMC, verbose)
% IGMCCLUSTER Incremental Grouped Mixture Clustering model (I-GMC) for Gaussian
%   mixture models. It is an interface for a C++ library that implements the an
%   incremental version of the GMC [1].
%
%   [IGMC, ischanged] = igmccluster (X, IGMC)
%   [IGMC, ischanged] = igmccluster (X, IGMC, verbose)
%
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the
%       number of dimensions.
%   - verbose true = print verbose output, false = no output. This is optional, 
%             default is false.
%
% Returns:
%   - gmm is the Gaussian Mixture Model structure. It has fields: TODO
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster.
%       .mu     [KxD] cluster means.
%       .sigma  [DxDxK] cluster covariances.
%   - ischanged [scalar] boolean if the model changed significantly.
%
% Notes:
%   - consistent scaling of X... TODO
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     22/07/2011
%
% References:
%   [1] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Hierarchal Bayesian
%       mixtures for clustering multiple related datasets." NIPS 2011
%       Submission, June 2011.
%
% See also VDPCLUSTER, GMCCLUSTER, GMMCLASSIFY, GMMPREDICT

    if nargout < 1, error('You need to at least have an I-GMC output!');  end
    if ~isfield(IGMC, 'N_s'), error('use an I-GMC structure!'); end

    % Convert I-GMC structure
    IGMC = convgmma2c(IGMC);

    % Run the suitable version of igmccluster_mex depending on the arguments
    if nargin == 2,
        [IGMC ischanged] = clusterinc_mex(X, IGMC);
    elseif nargin == 3,
        [IGMC ischanged] = clusterinc_mex(X, IGMC, logical(verbose));
    else
        error('Invalid number of input arguments.');
    end

    % Convert IGMC structure
    IGMC = convgmmc2a(IGMC);

end