function EMM = SS2EMM (SS)
% SS2EMM Sufficient statistics to Exponential Mixture Model structure conversion
%   The Exponential mixture returned is constructed using Maximum Likelihood
%   formulae.
%
%   EMM = SS2EMM (SS)
%
%   Arguments:
%
%   - SS is a sufficient statistics structure. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .priorval the value of the cluster hyperprior (e.g. cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[1xD]} array of observation suff. stats. of the mean
%       .ss2      {Kx[0x0]} empty (unrequired) sufficient statistic.
%
%   Returns:
%
%   - EMM is the Gaussian Mixture Model structure. It has fields:
%       .K      the number of clusters.
%       .w      [1xK] weights of each cluster in the entire model.
%       .lambda [KxD] cluster rates.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     04/04/2012
%
% See also SS2GMM, BMMCLUSTER, GMCCLUSTER

    N_k = cell2mat(SS.N_k);
    N   = sum(N_k);
    K   = SS.K;
    D   = size(SS.ss1{1},2);

    EMM.K     = K;
    EMM.w     = N_k/N;
    EMM.lamda = zeros(K,D);
    
    for k=1:K,
        EMM.lamda(k,:) = SS.N_k{k}./(SS.ss1{k} + eps); % avoid div by 0.
    end

end
