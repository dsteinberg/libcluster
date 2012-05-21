function [qZ, SSgroup, SS, F] = gmccluster (X, SS, SSgroup, options)
% GMCCLUSTER Groups of Mixtures Clustering model for Bayesian mixture models.  
%   This is an interface for a C++ library (libcluster) that implements various 
%   Bayesian Group-Mixture Models such as the GMC and SGMC [1] amongst others.
%
%   [qZ, SSgroup, SS, F] = gmccluster (X)
%   [qZ, SSgroup, SS, F] = gmccluster (X, SS)
%   [qZ, SSgroup, SS, F] = gmccluster (X, SS, SSgroup)
%   [qZ, SSgroup, SS, F] = gmccluster (X, SS, SSgroup, options)
%   [qZ, SSgroup, SS, F] = gmccluster (X, [], [], options)
%   [qZ, SSgroup, SS, F] = gmccluster (X, SS, [], options)
%   
% Inputs:
%
%   - X {Jx[NjxD]} cell array of observation/feature matrices. Nj is the number 
%       of elements in each group, j, D is the number of dimensions. This must 
%       be in the range [0, inf) for the BEMM algorithm.
%
%   - SS is a sufficient statistics structure for the model. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .D        the dimensionality of the suff. stats.
%       .priorval the value of the cluster hyperprior (cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%
%   - SSgroup {J X SS} cell array of group sufficient statistic structures.
%
%   - options specifies various algorithm options, they are:
%     * options.alg is the algorithm to use, valid options are:
%
%       + 'GMC'  Groups of Mixtures Clustering model with a Generalised
%                Dirichlet prior on the group weights (default).
%       + 'SGMC' Groups of Mixtures Clustering model with a Symmetric 
%                Dirichlet prior on the group weights.
%       + 'DGMC' Groups of Mixtures Clustering model with diagonal covariance 
%                and Generalised Dirichlet prior on the group weights.
%       + 'EGMC' Groups of Mixtures Clustering model with Exponential cluster
%                distributions (and generalised Dirichlet weights).
%
%     * options.sparse true = use sparse version of GMC (faster, less accurate), 
%                      false = use original, dense, GMC (default).
%     * options.verbose true = print verbose output, 0 = no output (default). 
%     * options.prior is the prior notion of the "shape" of the clusters. For 
%                     Gaussians this is the width of the clusters, for  
%                     Exponentials this is the  approximate magnitude of the 
%                     observations etc. The default value is 1e-5.
%     * options.nthreads is the number of threads to use. This is automatically 
%                        determined unless specified here.
%
% Returns (all are optional):
%   - qZ {Jx[NjxK]} probability of each observation belonging to each cluster.
%        This is the Variational posterior approximation to p(Z|X).
%   - SSgroup {J X SS} cell array of group sufficient statistic structures.
%   - SS is a sufficient statistics structure for the model as above.
%   - F [scalar] final free energy. 
%
% Notes:
%   - I find that standardising or whitening all of the dimensions of X before 
%     clustering with the Gaussian models improves results.
%   - The 'EGMC' algorithm is very sensitive to the prior, if in doubt, start it
%     with a value similar in magnitude as the data or greater.
%
% Author:   Daniel Steinberg
%           Australian Centre for Field Robotics
%           The University of Sydney
%
% Date:     4/04/2012
%
% References:
%   [1] D. M. Steinberg, O. Pizarro, and S. B. Williams, "Clustering Groups 
%       of Related Visual Datasets," unpublished, 2011.
%
% See also BMMCLUSTER, SS2GMM, SS2EMM

    % Check for a cell array
    if ~iscell(X), error('X must be a cell array!'); end
    
    % Check to see if X is double precision
    if isa( X{1}, 'double' ) == false
      error( 'X must be double precision' );
    end
    
    % Parse the options structure
    if nargin < 4,
      options = struct([]);
    end
    if isfield(options, 'sparse') == false,
      options(1).sparse = false;
    end
    if isfield(options, 'verbose') == false,
      options(1).verbose = false;
    end
    if isfield(options, 'prior') == false,
      options(1).prior = 1e-5;
    end
    if isfield(options, 'alg') == false,
      options(1).alg = 'gmc';
    end
    
  % Parse algorithm option
  switch lower(options.alg)
    case 'gmc'
        algval = 0;
    case 'sgmc',
        algval = 1;
    case 'dgmc',
        algval = 2;
    case 'egmc',
        algval = 3;
    otherwise
        error('Unknown algorithm specified!');
  end
  
  % Instantiate empty Suff stats if required
  if nargin < 2, SS = []; end
  if nargin < 3, SSgroup = []; end
    
  % Create an empty template SS structure if needed.
  if (isempty(SS) == true) || (isempty(SSgroup) == true),
    SStmp.priorval = options.prior;
    SStmp.K = 0;
    SStmp.D = size(X{1}, 2);
    SStmp.F = 0;
    SStmp.N_k = {};
    SStmp.ss1 = {};
    SStmp.ss2 = {};
  end
  
  % Check Model sufficient statistics
  if isempty(SS) == true,
  
    if isempty(SSgroup) == false,
      error('SS cannot be empty when being used with a populated SSgroup!');
    end
  
    SS = SStmp;
  end 
  
  % Check Group sufficient statistics  
  J = length(X); 
  if isempty(SSgroup) == true,
    
    SSgroup = cell(J, 1);
    for j = 1:J, SSgroup{j} = SStmp; end
    
  elseif any(size(X) ~= size(SSgroup)),
    error('X and SSgroup must have the same no. of cell elements');    
  end

  % Run the suitable version of clustergroup_mex depending on the arguments
  if isfield(options, 'nthreads') == false,
    [F qZ SSgroup SS] = clustergroup_mex(X, SS, SSgroup, algval, ...
                        logical(options.sparse), logical(options.verbose));
  else
    [F qZ SSgroup SS] = clustergroup_mex(X, SS, SSgroup, algval, ...
      logical(options.sparse), logical(options.verbose), options.nthreads);
  end
    
end
