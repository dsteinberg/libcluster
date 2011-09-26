function [Z, qZ, wj] = igmcclassify (X, IGMC, verbose)
% IGMCCLASSIFY Classify data using the Incremental Grouped Mixture Clustering
%   model (I-GMC) for Gaussian mixture models. It is an interface for a C++
%   library that implements the an incremental version of the GMC [1].
%
%   [Z, qZ, wj] = igmcclassify (X, IGMC)
%   [Z, qZ, wj] = igmcclassify (X, IGMC, verbose)
%
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the
%       number of dimensions.
%   - verbose 1 = print verbose output, 0 = no output. This is optional, default
%             is 0.
%
% Returns:
%   - TODO
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

    if ~isfield(IGMC, 'Nk_'), error('use an IGMC structure!'); end

    % Convert I-GMC structure
    IGMC = convgmma2c(IGMC);

    % Run the suitable version of igmccluster_mex depending on the arguments
    if nargin == 2,
        [qZ wj] = igmcclassify_mex(X, IGMC);
    elseif nargin == 3,
        [qZ wj] = igmcclassify_mex(X, IGMC, logical(verbose));
    else
        error('Invalid number of input arguments.');
    end

    % Get hard assignments
    [tmp, Z] = max(qZ,[],2);

end