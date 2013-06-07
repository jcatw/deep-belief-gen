function [ dbn ] = dbntrain( x, L, T, B, C, K, alpha, lambda )
%DBNTRAIN Train a deep belief network.
%   Train a deep belief network (DBN) via greedy stochastic mini-batch
%   gradient descent.
%   
%   Arguments:
%       x: training data
%       L: number of hidden layers
%       T: number of training iterations per hidden layer
%       B: vector of length L with the 
%          number of batches per training iteration
%       C: number of parallel chains to use for the monte carlo 
%          approximation to the likelihood gradient.
%       K: vector of length L with the number of hidden variables per layer
%       alpha: learning rate
%       lambda: regularization parameter
%
%   Returns:
%   Structure dbn
%       dbn.rbms: The rbms which compose the dbn
%                 (length L cell array of rbm structures)  
%       dbn.L: Number of hidden layers
%       dbn.K: Vector of hidden layer sizes
%       dbn.D: Size of visible layer

%   James Atwood 6/4/13
%   jatwood@cs.umass.edu

assert(length(K) == L,'Wrong hidden unit length');
assert(length(B) == L,'Wrong batch length');

[N,D] = size(x);

%dbn.rbms = [];
dbn.L = L;
dbn.K = K;
dbn.D = D;

for i=1:L
    transformed_x = x;
    for j=1:(i-1)
        transformed_x = rbmup(dbn.rbms{j},transformed_x);
    end
    dbn.rbms{i} = rbmtrain(transformed_x, T, B(i), C, K(i), alpha, lambda);
end

% untie parameters
for i=1:length(dbn.rbms)-1
    dbn.rbms{i}.gen.Wb = dbn.rbms{i}.Wb;
    dbn.rbms{i}.gen.Wc = dbn.rbms{i}.Wc;
    dbn.rbms{i}.gen.Wp = dbn.rbms{i}.Wp;

    dbn.rbms{i}.rec.Wb = dbn.rbms{i}.Wb;
    dbn.rbms{i}.rec.Wc = dbn.rbms{i}.Wc;
    dbn.rbms{i}.rec.Wp = dbn.rbms{i}.Wp;
end

end

