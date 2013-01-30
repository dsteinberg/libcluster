% The Symmetric Grouped Mixtures Clustering (S-GMC) clustering algorithm.  This
%   function uses the Symmetric Grouped Mixtures Clustering model [1] to 
%   cluster multiple datasets simultaneously with cluster sharing between 
%   datasets. It uses a symmetric Dirichlet prior over the group mixture 
%   weights, and a Gaussian-Wishart prior over the cluster parameters.
%
%  [qZ, weights, means, covariances] = learnSGMC (X, options)
%
% Arguments:
%  - X, {Jx[NxD]} cell array of observation matrices (one cell for each group)
%  - options, structure with members (all are optional):
%     + prior, [double] prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + sparse, [bool] do fast but approximate sparse VB updates (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qZ, {Jx[NxK]} cell array of assignment probablities
%  - weights, {Jx[1xK]} Group mixture weights
%  - means, {Kx[1xD]} Gaussian mixture means
%  - covariances, {Kx[DxD]} Gaussian mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   27/09/2012
%
% References:
%  [1] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
%      Thesis, 2012.

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
