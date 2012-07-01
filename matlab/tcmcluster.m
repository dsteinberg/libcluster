function [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, SS, SSdocs, options)
% TCMCLUSTER Topic clustering model for clustering Bayesian topic models.  
%   This is an interface for a C++ library (libcluster) that implements various 
%   Bayesian Topic models such as the TCM amongst others.
%
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T)
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, SS)
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, SS, SSdocs)
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, SS, SSdocs, options)
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, [], [], options)
%   [qY, qZ, SSdocs, SS, clparams, F] = tcmcluster (X, T, SS, [], options)
%   
% Inputs:
%
%   - X {Ix[NixD]} cell array of observation/feature matrices. Ni is the number 
%       of elements in each document, i, D is the number of dimensions.
%
%   - SS is a sufficient statistics structure for the model. It has fields:
%       .K        the number of clusters for which there are suff. stats.
%       .D        the dimensionality of the suff. stats.
%       .priorval the value of the cluster hyperprior (cluster width)
%       .N_k      {1xK} the number of observations in each cluster.
%       .ss1      {Kx[?x?]} array of observation suff. stats. no 1.
%       .ss2      {Kx[?x?]} array of observation suff. stats. no 2.
%
%   - SSdocs {I X SS} cell array of document sufficient statistic structures.
%
%   - T [integer] truncation level of classes, i.e. max number of classes to 
%          find.
%
%   - options specifies various algorithm options, they are:
%     * options.alg is the algorithm to use, valid options are:
%
%       + 'TCM' Topic clustering model with Dirichlet weights and class 
%               parameters, and Gaussian-Wishart clusters. 
%
%     * options.sparse true = use sparse version of TCM (faster, less accurate), 
%                      false = use original, dense, TCM (default).
%     * options.verbose true = print verbose output, 0 = no output (default). 
%     * options.prior is the prior notion of the "shape" of the clusters. For 
%                     Gaussians this is the width of the clusters, for  
%                     Exponentials this is the  approximate magnitude of the 
%                     observations etc. The default value is 1e-5.
%     * options.nthreads is the number of threads to use. This is automatically 
%                        determined unless specified here.
%
% Returns (all are optional):
%   - qY [I x T] probability of each document, i, belonging to a class, t. 
%   - qZ {Ix[NixK]} probability of each word observation belonging to a cluster.
%        This is the Variational posterior approximation to p(Z|X).
%   - SSdocs {I X SS} cell array of document sufficient statistic structures.
%   - SS is a sufficient statistics structure for the model as above.
%   - clparams [T x K] the class parameters, in each row, t.
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
% See also GMCCLUSTER, BMMCLUSTER, SS2GMM, SS2EMM

    % Check for valid truncation levels
    if T < 0, error('T must be greater than or equal to zero!'); end

    % Check for a cell array
    if ~iscell(X), error('X must be a cell array!'); end
    
    % Check to see if X is double precision
    if isa(X{1}, 'double') == false
      error('X must be double precision');
    end
    
    % Parse the options structure
    if nargin < 5,
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
      options(1).alg = 'tcm';
    end
    
  % Parse algorithm option
  switch lower(options.alg)
    case 'tcm'
        algval = 0;
    otherwise
        error('Unknown algorithm specified!');
  end
  
  % Instantiate empty Suff stats if required
  if nargin < 3, SS = []; end
  if nargin < 4, SSdocs = []; end
    
  % Create an empty template SS structure if needed.
  if (isempty(SS) == true) || (isempty(SSdocs) == true),
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
  
    if isempty(SSdocs) == false,
      error('SS cannot be empty when being used with a populated SSdocs!');
    end
  
    SS = SStmp;
  end 
  
  % Check Group sufficient statistics  
  I = length(X); 
  if isempty(SSdocs) == true,
    
    SSdocs = cell(I, 1);
    for i = 1:I, SSdocs{i} = SStmp; end
    
  elseif any(size(X) ~= size(SSdocs)),
    error('X and SSdocs must have the same no. of cell elements');    
  end

  % Run the suitable version of topic_mex depending on the arguments
  if isfield(options, 'nthreads') == false,
    [F qY qZ SSdocs SS clparams] = topic_mex(X, SS, SSdocs, T, algval, ...
                             logical(options.sparse), logical(options.verbose));
  else
    [F qY qZ SSdocs SS clparams] = topic_mex(X, SS, SSdocs, T, algval, ...
           logical(options.sparse), logical(options.verbose), options.nthreads);
  end
    
end
