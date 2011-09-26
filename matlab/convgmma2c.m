function gmm = convgmma2c (gmm)
% CONVGMMC2A Convert a GMM struct from having ND arrays to cell arrays. The
%   reason this is not done in C++ is the mex library makes it... hard!
%
%   gmm = convgmma2c(gmm)
%
%   Input:      A GMM with ND array fields.
%   Returns:    A GMM with Cell array fields for w {1xK scalar}, mu {1xK [1xD]} 
%               and sigma {1xK [DxD]}.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     21/07/2011

        K = gmm.K;
        
        if (size(gmm.w,2) ~= K) || (size(gmm.mu,1) ~= K) ...
            || ((size(gmm.sigma,3) ~= K) && (K ~= 0)) ...
            || (size(gmm.mu,2) ~= size(gmm.sigma,1)) ...
            || (size(gmm.mu,2) ~= size(gmm.sigma,2)),
            error('The GMM properties are not consistent!');
        end

        w     = cell(1, K);
        mu    = cell(1, K);
        sigma = cell(1, K);
        for k = 1:K,
            w{k}     = gmm.w(k); 
            mu{k}    = gmm.mu(k,:);
            sigma{k} = gmm.sigma(:,:,k);
        end
        gmm.w = w;
        gmm.mu = mu;
        gmm.sigma = sigma;

end