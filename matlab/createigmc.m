function IGMC = createigmc (J, D, kappa, tau0, cmeanp, cwidthp)
%
% TODO
%

% Check number arguments, input defaults
if nargin < 4 || nargin > 6, error('wrong number of inputs!'); end
if nargin < 5, cmeanp = zeros(1,D); end
if nargin < 6, cwidthp = 1; end

% Check arguments
if kappa < 0, error('kappa must be greater than or equal to 0!'); end
if tau0 < 1, error('tau0 must be greater than or equal to 1!'); end
if J < 1, error('J must be greater than or equal to 1!'); end
if D < 1, error('D must be greater than or equal to 1!'); end
if cwidthp <= 0, error('cwidthp must be greater than 0!'); end
if any(size(cmeanp) ~= [1, D]), error('size(cmeanp) must be [1, D]!'); end

IGMC.J = J;
IGMC.D = D;
IGMC.kappa = kappa;
IGMC.tau0 = tau0;
IGMC.cwidthp = cwidthp;
IGMC.cmeanp = cmeanp;

IGMC.K = 0;
IGMC.tau = 1;
IGMC.lambda = 0;
IGMC.rho = 1;
IGMC.Fpi = 0;
IGMC.Fxz = 0;

IGMC.w  = [];
IGMC.mu = [];
IGMC.sigma = [];

IGMC.Nk_ = {};
IGMC.Xk_ = {};
IGMC.Rk_ = {};

end