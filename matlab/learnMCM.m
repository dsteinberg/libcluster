% The Multiple Clustering Model (MCM) clustering algorithm.
%
%  This function uses the Multiple Clustering Model [1] to simultaneously
%  cluster words/segment observations and documents/image observations while
%  also sharing these between datasets. It uses a Generalised Dirichlet prior
%  over the group mixture weights, a Dirichlet prior over the document/image
%  proportions, and a Gaussian-Wishart prior over the word/segment and
%  document/image cluster parameters.
%
%  [qY, qZ, iweights, sweights, imeans, icovariances, smeans, scovariances] 
%     = learnMCM (W, X, options)
%
% Arguments:
%  - W, {Jx[IjxD1]} cells of document/image observation matrices, j over the 
%       groups, Ij over the documents/images.
%  - X, {Jx{Ijx[NijxD]}} nested cells of  word/segment observation matrices, j
%       over the groups, Ij over the documents/images, Nij over the 
%       words/segments.
%  - options, structure with members (all are optional):
%     + trunc, [unsigned int] the max number of image clusters to find 
%              (100 default)
%     + prior, [double] prior cluster value (1 default)
%     + prior2, [double] prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + sparse, [bool] do fast but approximate sparse VB updates (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qY, {Jx[IjxT]} cell array of image to document/image cluster assignments
%  - qZ, {Jx{Ijx[NijxK]}} nested cell array of word/segment cluster assignments
%  - iweights, {Jx[1xK]} Group document/image cluster weights
%  - sweights, [TxK] Dirichlet (document/image cluster) segment proportions per
%        image/document cluster.
%  - imeans, {Kx[1xD]} Gaussian (document/image cluster) mixture means
%  - icovariances, {Kx[DxD]} Gaussian (document/image cluster) mixture 
%        covariances
%  - smeans, {Kx[1xD]} Gaussian (word/segment cluster) mixture means
%  - scovariances, {Kx[DxD]} Gaussian (word/segment cluster) mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   23/11/2012
%
% References:
%  [1] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
%      Thesis, 2012.
