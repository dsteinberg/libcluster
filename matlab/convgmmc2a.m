function gmm = convgmmc2a (gmm)
% CONVGMMC2A Convert a GMM struct from having cell arrays to ND arrays. The
%   reason this is not done in C++ is the mex library makes it... hard!
%
%   gmm = convgmmc2a(gmm)
%
%   Input:      A GMM with cell array fields.
%   Returns:    A GMM with ND array fields for w [1xK], mu [KxD] and sigma
%               [DxDxK].
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     21/07/2011

        K = gmm.K;
        D = size(gmm.mu{1},2);

        w     = zeros(1,K);
        mu    = zeros(K,D);
        sigma = zeros(D,D,K);
        for k = 1:K,
            w(k)         = gmm.w{k}; 
            mu(k,:)      = gmm.mu{k};
            sigma(:,:,k) = gmm.sigma{k};
        end
        gmm.w = w;
        gmm.mu = mu;
        gmm.sigma = sigma;

end