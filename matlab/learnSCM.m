% The Simulaneous Clustering Model (SCM) clustering algorithm.
%  This function uses the Simultaneous Clustering Model [1] to simultaneously 
%  cluster words/segments and documents/images while also sharing these between 
%  datasets. It uses a Generalised Dirichlet prior over the group mixture 
%  weights, a Dirichlet prior over the document/image clusters, and a 
%  Gaussian-Wishart prior over the word/segment cluster parameters.
%
%  [qY, qZ, weights, classes, means, covariances] = learnSCM (X, options)
%
% Arguments:
%  - X, {Jx{Ijx[NijxD]}} nested cells of observation matrices, j over the 
%       groups, Ij over the documents/images, Nij over the words/segments.
%  - options, structure with members (all are optional):
%     + trunc, [unsigned int] the max number of classes to find (100 default)
%     + prior, [double] prior cluster value (1e-5 default)
%     + verbose, [bool] verbose output flag (false default)
%     + sparse, [bool] do fast but approximate sparse VB updates (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qY, {Jx[IjxT]} cell array of image to document/image cluster assignments
%  - qZ, {Jx{Ijx[NijxK]}} nested cell array of word/segment cluster assignments
%  - weights, {Jx[1xK]} Group document/image cluster weights
%  - classes, [TxK] Dirichlet (document/image cluster) parameters (weights)
%  - means, {Kx[1xD]} Gaussian (word/segment cluster) mixture means
%  - covariances, {Kx[DxD]} Gaussian (word/segment cluster) mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   16/08/2012
%
% References:
%  [1] ???
