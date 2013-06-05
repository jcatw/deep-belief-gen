function [ samples ] = dbnsample( dbn, N, t )
%DBNSAMPLE Sample from a deep belief network
%   Sample N items from a deep belief network.
%
%   Arguments:
%       dbn: A deep belief network structure
%       N: The number of samples to generate
%       t: The number of alternating Gibbs sampling steps to take at
%          the top-level rbm
%
%   Returns:
%       samples: An N x dbn.D matrix of samples

samples = zeros(N,dbn.D);

top_rbm = dbn.rbms{end};

for i=1:N
    s = rbmsample(top_rbm,1,t,1);
    for j=(dbn.L-1):-1:1
        s = rbmdown(dbn.rbms{j},s);
    end
    samples(i,:) = s;
end
end

