function [ x ] = rbmsample( rbm, N, burn_in, interval )
%RBMSAMPLE Generate n samples from a restricted Boltzmann machine (RBM).
%   Generate n samples from an RBM via alternating Gibbs sampling.
%
%   Arguments:
%       rbm: rbm structure
%       N: number of samples to generate
%       burn_in: number of ignored initial iterations
%       interval: after burn in, record once per interval samples
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

h = zeros(N,K);
x = zeros(N,D);

%h(1,:) = randi(2,1,K) - 1;
last_h = randi(2,1,K) - 1;

n = 0;
i = 0;
while n<N
    i = i + 1;
    %prob_x = exp(rbm.Wc + (rbm.Wp * h(i,:)')') ./ (1 + exp(rbm.Wc + (rbm.Wp * h(i,:)')'));
    prob_x = exp(rbm.Wc + (rbm.Wp * last_h')') ./ (1 + exp(rbm.Wc + (rbm.Wp * last_h')'));
    last_x = prob_x > rand(1,D);
    %x(i,:) = prob_x > rand(1,D);
    
    %prob_h = exp(rbm.Wb' + (x(i,:) * rbm.Wp)) ./ (1 + exp(rbm.Wb' + (x(i,:) * rbm.Wp)));
    prob_h = exp(rbm.Wb' + (last_x * rbm.Wp)) ./ (1 + exp(rbm.Wb' + (last_x * rbm.Wp)));
    last_h = prob_h > rand(1,K);
    %h(i+1,:) = prob_h > rand(1,K);
    
    if i > burn_in && mod(i - burn_in, interval) == 0
        n = n + 1;
        x(n,:) = last_x;
        h(n,:) = last_h;
    end
end
end