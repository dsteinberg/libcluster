function GMM = SS2GMM (SS)
% SS2GMM Sufficient statistics to Gaussian Mixture Model structure conversion.
%   The Gaussian mixture returned is constructed using Maximum Likelihood
%   formulae.
%
%   GMM = SS2GMM (SS)
%
%   Arguments:
%
%   - SS is a sufficient statistics structure. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .priorval the value of the cluster hyperprior (e.g. cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[1xD]} array of observation suff. stats. of the mean
%       .ss2      {Kx[DxD]} or {Kx[1xD]} array of observation suff. stats. of 
%                 the covariance or variance (diagonal)
%
%   Returns:
%
%   - GMM is the Gaussian Mixture Model structure. It has fields:
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster in the entire model.
%       .mu     [KxD] cluster means.
%       .sigma  [DxDxK] cluster covariances.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     8/12/2011

    N_k = cell2mat(SS.N_k);
    N   = sum(N_k);
    K   = SS.K;
    D   = size(SS.ss1{1},2);
    isdiag = size(SS.ss2{1},1) == 1;

    GMM.K     = K;
    GMM.w     = N_k/N;
    GMM.mu    = zeros(K,D);
    GMM.sigma = zeros(D,D,K);
    
    for k=1:K,
        GMM.mu(k,:) = SS.ss1{k}/SS.N_k{k};
        
        if isdiag == true,
          GMM.sigma(:,:,k) = diag(SS.ss2{k}/SS.N_k{k} - GMM.mu(k,:).^2);
        else
          GMM.sigma(:,:,k) = SS.ss2{k}/SS.N_k{k} - GMM.mu(k,:)'*GMM.mu(k,:);
        end
        
    end

end
