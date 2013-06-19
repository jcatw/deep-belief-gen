function [ v ] = rbmdown( rbm, h )
%RBMUP Sample visible units given hidden
%   Given an rbm structure and a matrix of hidden units, return a matrix
%   of visible units.

% James Atwood, 6/5/13

[N,dummy] = size(h);

D = length(rbm.Wc);

v = zeros(N,D);

for n=1:N
    this_h = h(n,:);
    prob_v = exp(rbm.Wc + (rbm.Wp * this_h')') ./ (1 + exp(rbm.Wc + (rbm.Wp * this_h')'));
    v(n,:) = prob_v > rand(1,D);
end

end

