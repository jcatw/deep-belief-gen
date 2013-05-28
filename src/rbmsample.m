function [ x ] = rbmsample( rbm, n )
%RBMSAMPLE Generate n samples from a restricted Boltzmann machine (RBM).
%   Generate n samples from an RBM via alternating Gibbs sampling.
%
%   Arguments:
%       rbm: rbm structure
%       n: number of samples to generate
%
%   Returns:
%       x: matrix of samples
%
%   Example:
%   rbm = rbmtrain(MNISTX, 50, 100, 100, 400, 0.1, 0.0001);
%   x = rbmsample(rbm, 100);

% James Atwood 5/27/13
% jatwood@cs.umass.edu

[D,K] = size(rbm.Wp);

h = zeros(n+1, K);
x = zeros(n,D);

h(1,:) = randi(2,1,K) - 1;

for i=1:n
    prob_x = exp(rbm.Wc + (rbm.Wp * h(i,:)')') ./ (1 + exp(rbm.Wc + (rbm.Wp * h(i,:)')'));
    x(i,:) = prob_x > rand(1,D);
    
    prob_h = exp(rbm.Wb' + (x(i,:) * rbm.Wp)) ./ (1 + exp(rbm.Wb' + (x(i,:) * rbm.Wp)));
    h(i+1,:) = prob_h > rand(1,K);
end
end