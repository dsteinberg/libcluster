% The Multiple-source Clustering Model (MCM) clustering algorithm.
%  This function uses the Multiple-source Clustering Model [1, 2] to 
%  simultaneously cluster words/segment observations and documents/image 
%  observations while also sharing these between datasets. It uses a Generalised
%  Dirichlet prior over the group mixture weights, a Dirichlet prior over the
%  document/image cluster proportion parameters, and Gaussian-Wishart priors
%  over the word/segment and document/image cluster parameters.
%
%  [qY, qZ, weights_j, weights_t, means_t, covariances_t, means_k, 
%       covariances_k] = learnMCM (W, X, options)
%
% Arguments:
%  - W, {Jx[IjxDt]} cells of document/image observation matrices, j over the 
%       groups, Ij over the documents/images.
%  - X, {Jx{Ijx[NijxDb]}} nested cells of  word/segment observation matrices, j
%       over the groups, Ij over the documents/images, Nij over the 
%       words/segments.
%  - options, structure with members (all are optional):
%     + trunc, [unsigned int] the max number of top-level (document/image) 
%              clusters to find (100 default)
%     + prior, [double] top-level prior cluster value (1 default)
%     + prior2, [double] bottom-level prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qY, {Jx[IjxT]} cell array of image to document/image cluster assignments
%  - qZ, {Jx{Ijx[NijxK]}} nested cell array of word/segment cluster assignments
%  - weights_j, {Jx[1xK]} Group document/image cluster weights
%  - weights_t, [TxK] Dirichlet word/segment proportions per image/document
%        cluster (top-level cluster proportion parameters).
%  - means_t, {Tx[1xDt]} Gaussian (document/image cluster) mixture means
%  - covariances_t, {Tx[DtxDt]} Gaussian (document/image cluster) mixture 
%        covariances
%  - means_k, {Kx[1xDb]} Gaussian (word/segment cluster) mixture means
%  - covariances_k, {Kx[DbxDb]} Gaussian (word/segment cluster) mixture
%        covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   19/10/2013
%
% References:
%  [1] Synergistic Clustering of Image and Segment Descriptors for Unsupervised
%      Scene Understanding. D. M. Steinberg, O. Pizarro, S. B. Williams. In 
%      International Conference on Computer Vision (ICCV). IEEE, Sydney, NSW, 
%      2013.
%  [2] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
%      Thesis, 2013.

% libcluster -- A collection of Bayesian clustering algorithms
% Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
