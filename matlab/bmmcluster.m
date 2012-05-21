function [Z, qZ, SS, F] = bmmcluster (X, SS, options)
% BMMCLUSTER Bayesian Mixture models. 
%   This is an interface for a C++ library (libcluster) that implements various 
%   Bayesian Mixture Models such as the Bayesian GMM [1] and Variational 
%   Dirichlet process [2], amongst others.
% 
%   [Z, qZ, SS, F] = bmmcluster (X)
%   [Z, qZ, SS, F] = bmmcluster (X, SS)
%   [Z, qZ, SS, F] = bmmcluster (X, SS, options)
%   [Z, qZ, SS, F] = bmmcluster (X, [], options)
%   
% Inputs:
%   - X [NxD] observation/feature matrix. N is the number of elements, D is the 
%       number of dimensions. This must be in the range [0, inf) for the BEMM 
%       algorithm.
%
%   - SS is a sufficient statistics structure for the model. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .D        the dimensionality of the suff. stats.
%       .priorval the value of the cluster hyperprior (cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%
%   - options specifies various algorithm options, they are:
%
%     * options.alg is the algorithm to use, valid inputs are:
%
%       + 'VDP'  which is the Variational Dirichlet Process for GMMs (default)
%       + 'BGMM' which is the Bayesian Gaussian Mixture Model
%       + 'DGMM' which is the Bayesian Gaussian Mixture Model with diagonal 
%                covariance clusters
%       + 'BEMM' which is the Bayesian Exponential Mixture Model
%
%     * options.verbose true = print verbose output, false = no output (default) 
%     * options.prior is the prior notion of the "shape" of the clusters. For  
%                     Gaussians this is the width of the clusters, for 
%                     Exponentials this is the approximate magnitude of the 
%                     observations etc. The default value is 1e-5.
%     * options.nthreads is the number of threads to use. This is automatically  
%                        determined unless specified here.
%
% Returns (all are optional):
%   - Z [Nx1] labels. These are the most likely clusters for each observation
%       in X.
%   - qZ [NxK] probability of each observation belonging to each cluster. This 
%        is the Variational posterior approximation to p(Z|X).
%   - SS is a sufficient statistics structure. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .priorval the value of the cluster hyperprior (e.g. cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%   - F [scalar] final free energy. 
%
% Notes:
%   - The difference between the VDP and GMM algorithms is that the prior over
%     the mixture weights is different. The VDP uses a Stick-Breaking prior,
%     while the GMM uses a symmetric Dirichlet prior.
%   - I find that standardising or whitening all of the dimensions of X before 
%     clustering improves results with the Gaussian algorithms. 
%   - The 'BEMM' algorithm is very sensitive to the prior, if in doubt, start it
%     with a value similar in magnitude as the data or greater. 
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     4/04/2012
%
% References:
%   [1] C. M. Bishop, Pattern Recognition and Machine Learning. Cambridge, UK:
%       Springer Science+Business Media, 2006.
%
%   [2] K. Kurihara, M. Welling, and N. Vlassis, Accelerated variational 
%       Dirichlet process mixtures, Advances in Neural Information 
%       Processing Systems, vol. 19, p. 761, 2007.
%
% See also GMCCLUSTER, SS2GMM, SS2EMM

  % Check to see if X is double precision
  if isa( X, 'double' ) == false
    error( 'X must be double precision' );
  end
  
  % Parse the options structure
  if nargin < 3,
    options = struct([]);
  end
  if isfield(options, 'verbose') == false
    options(1).verbose = false;
  end
  if isfield(options, 'prior') == false
    options(1).prior = 1e-5;
  end
  if isfield(options, 'alg') == false
    options(1).alg = 'vdp';
  end
  
  % Parse Alg argument
  switch lower(options.alg)
      case 'vdp'
          algval = 0;
      case 'bgmm',
          algval = 1;
      case 'dgmm'
          algval = 2;
      case 'bemm',
          algval = 3;
      otherwise
          error('Unknown algorithm specified!');
  end
  
  % Instantiate empty Suff stats if required
  if nargin < 2, SS = []; end
  
  % Create an empty SS structure if needed.
  if (isempty(SS)) == true,
    SS.priorval = options.prior;
    SS.K = 0;
    SS.D = size(X, 2);
    SS.F = 0;
    SS.N_k = {};
    SS.ss1 = {};
    SS.ss2 = {};
  end
          
  % Run the suitable version of cluster_mex depending on the arguments
  if isfield(options, 'nthreads') == false,
    [F qZ SS] = cluster_mex(X, SS, algval, logical(options.verbose));
  else
    [F qZ SS] = cluster_mex(X, SS, algval, logical(options.verbose), ...
                            options.nthreads);
  end
  
  % Find most likely qZ and assign it to z 
  [~, Z] = max(qZ,[],2);
  
end
