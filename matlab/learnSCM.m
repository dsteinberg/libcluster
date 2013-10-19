% The Simulaneous Clustering Model (SCM) clustering algorithm.
%  This function uses the Simultaneous Clustering Model [1] to simultaneously
%  cluster words/segments etc and documents/images etc while also sharing these
%  between datasets. It uses a Generalised Dirichlet prior over the group
%  mixture weights, a Dirichlet prior over the document/image clusters, and a
%  Gaussian-Wishart prior over the word/segment cluster parameters.
%
%  [qY, qZ, weights_j, weights_t, means, covariances] = learnSCM (X, options)
%
% Arguments:
%  - X, {Jx{Ijx[NijxD]}} nested cells of observation matrices, j over the 
%       groups, Ij over the documents/images, Nij over the words/segments.
%  - options, structure with members (all are optional):
%     + trunc, [unsigned int] the max number of image clusters to find 
%       (100 default)
%     + prior, [double] top-level (Dirichlet) prior cluster value (1 default)
%     + prior2, [double] bottom-level prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + sparse, [bool] do fast but approximate sparse VB updates (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qY, {Jx[IjxT]} cell array of image to document/image cluster assignments
%  - qZ, {Jx{Ijx[NijxK]}} nested cell array of word/segment cluster assignments
%  - weights_j, {Jx[1xK]} Group document/image (top-level) cluster weights
%  - weights_t, [TxK] Dirichlet document/image cluster parameters. These are
%       also bottom level-cluster (word/segment) proportions
%  - means, {Kx[1xD]} Gaussian (word/segment cluster) mixture means
%  - covariances, {Kx[DxD]} Gaussian (word/segment cluster) mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   19/10/2013
%
% References:
%  [1] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
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
