% The Bayseian Gaussian Mixture Model (BGMM) clustering algorithm.
%  This function uses Bayesian Gaussian Mixture model [2] to cluster data. It is
%  very similar to a Variational Dirichlet Process [1], but uses a Dirichlet
%  prior over the model weights.
%
%  [qZ, weights, means, covariances] = learnBGMM (X, options)
%
% Arguments:
%  - X, [NxD] observation matrix
%  - options, structure with members (all are optional):
%     + prior, [double] prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qZ, [NxK] assignment probablities
%  - weights, [1xK] Gaussian mixture weights
%  - means, {Kx[1xD]} Gaussian mixture means
%  - covariances, {Kx[DxD]} Gaussian mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   13/08/2012
%
% References:
%  [1] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational
%     Dirichlet process mixtures, Advances in Neural Information Processing
%     Systems, vol. 19, p. 761, 2007.
%  [2] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
%      Springer Science+Business Media, 2006.

% libcluster -- A collection of hierarchical Bayesian clustering algorithms.
% Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
%
% This file is part of libcluster.
%
% libcluster is free software: you can redistribute it and/or modify it under
% the terms of the GNU Lesser General Public License as published by the Free
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
%
% libcluster is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
% FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
% for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with libcluster. If not, see <http://www.gnu.org/licenses/>.
