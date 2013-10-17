function [dn] = dependency_network_fit(X,alpha)
%DEPENDENCY_NETWORK_FIT Fit a dependency network to X with inclusion tolerance alpha.
%   Fit a dependency network to X, a binary matrix, using logistic regression.  Features are
%   included if the magnitude of their coefficient is greater than or equal to alpha.
%
%   Arguments:
%       X: Binary matrix.  Rows are instances, columns features.
%       alpha: feature inclusion tolerance.
%
%   Returns:
%       dn: a dependency network structure

%   James Atwood 8/30/13
%   jatwood@cs.umass.edu

[ninstances, nfeatures] = size(X);

dn.network = zeros(nfeatures,nfeatures);
dn.coefficients = zeros(nfeatures, nfeatures-1);

for i=1:nfeatures
    if mod(i,10) == 0 || i == 1
      fprintf(1,'DN iteration %d\n',i);
    end
    %size([X(:,1:(i-1)),X(:,(i+1):end)])
    %size(X(:,i))
    [b, included, dev, stats] = logistic_regression( [X(:,1:(i-1)),X(:,(i+1):end)], ...
						     X(:,i), ...
						     alpha ); 
    %'included'
    %size(included)
    %'dn.network'
    %size(dn.network)
    %'dn.network(:,(i+1):end)'
    %size(dn.network((i+1):end),i)
    %'included(i:end)'
    %size(included(i:end))
    dn.network(1:(i-1),i) = double(included(1:(i-1)));
    dn.network((i+1):end,i) = double(included(i:end));
    dn.coefficients(i,:) = b;
end

	 
