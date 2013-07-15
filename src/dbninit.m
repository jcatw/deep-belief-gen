function [ dbn ] = dbninit(N, B, K, D, L, lablen)
%DBNINIT Initialize a deep belief network structure.  Should only be called by other functions.

%   James Atwood 7/15/13
%   jatwood@cs.umass.edu

assert(L > 0, 'Need at least one hidden layer');
assert(length(K) == L,'Wrong hidden unit length');

assert(mod(N,B) == 0,sprintf('Batch size B (currently %d) must be a factor of the number of training instances (%d)',B,N));

dbn.Nb = N / B;


% parameters and pre-initialization
dbn.L = L;
dbn.K = K;
dbn.D = D;
if(nargin > 4)
  dbn.numclasses = lablen;
end

dbn.rec.bias = cell(1,L+1);  % work with receptive weights, set generative to result when done
dbn.rec.pair = cell(1,L);

if(nargin > 5)
  dbn.rec.label.pair = sparse(0.1 * randn(lablen,K(end)));
  dbn.rec.label.bias = sparse(0.1 * randn(1,lablen));
end


dbn.rec.bias{1} = sparse(0.1 * randn(1,D));
dbn.rec.pair{1} = sparse(0.1 * randn(D,K(1)));
for i=2:L
  dbn.rec.bias{i} = sparse(0.1 * randn(1,K(i-1)));
  dbn.rec.pair{i} = sparse(0.1 * randn(K(i-1),K(i)));
end
dbn.rec.bias{L+1} = sparse(0.1 * randn(1,K(L)));
