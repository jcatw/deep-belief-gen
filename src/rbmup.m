function [ h ] = rbmup( rbm, v )
%RBMUP Sample hidden units given visible
%   Given an rbm structure and a matrix of visible units, return a matrix
%   of hidden units.

% James Atwood, 6/5/2013

[N, dummy] = size(v);
K = length(rbm.Wb);

h = zeros(N,K);

for n=1:N
    prob_h = exp(rbm.Wb' + (v(n,:) * rbm.Wp)) ./ (1 + exp(rbm.Wb' + (v(n,:) * rbm.Wp)));
    h(n,:) = prob_h > rand(1,K);
end
end

