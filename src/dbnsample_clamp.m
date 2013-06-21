function [ x ] = dbnsample_clamp( dbn, label, N, G, burn_in )
%DBNSAMPLE Sample from a deep belief network
%   Sample N items from a deep belief network.
%
%   Arguments:
%       dbn: A deep belief network structure.
%       N: The number of samples to generate.
%       G: The number of alternating Gibbs sampling steps to take at
%          the top-level rbm.
%       burn_in: the number of initial samples to discard.  A single sample is generated
%                every G iterations, so the total number of extra iterations introduced is
%                burn_in * G.
%
%   Returns:
%       samples: An N x dbn.D matrix of samples.

samples = zeros(N,dbn.D);
labels = repmat(label,N,1);

% init to random binary visible state and sample to penultimate layer
#s = randi(2,1,dbn.D) - 1;
#s = init;
#for k=1:dbn.L-1
#  p = logistic(s * dbn.rec.pair{k} + dbn.rec.bias{k+1});
#  s = p > rand(1,dbn.K(k));
#end

% burn-in the top level
%  half-sampling hack
prob_h = logistic(((dbn.numclasses + dbn.K(end-1))/dbn.numclasses) * labels * dbn.rec.label.pair + repmat(dbn.rec.bias{dbn.L+1},N,1));
h = prob_h > rand(N,dbn.K(end));
prob_s = logistic(h * dbn.gen.pair{dbn.L} + repmat(dbn.gen.bias{dbn.L},N,1)  );
s = prob_s > rand(N,dbn.K(end-1));
for b=1:burn_in
  for j=1:G
    prob_h = logistic(s * dbn.rec.pair{dbn.L} + ...
		      labels * dbn.rec.label.pair + ...
		      repmat(dbn.rec.bias{dbn.L+1},N,1));
    h = prob_h > rand(N,dbn.K(end));
    prob_s = logistic(h * dbn.gen.pair{dbn.L} + repmat(dbn.gen.bias{dbn.L}  ));
    s = prob_s > rand(N,dbn.K(end-1));
  end
end

% perform alternating gibbs sampling with a down pass every G iterations
for j=1:G
  prob_h = logistic(s * dbn.rec.pair{dbn.L} + ...
		    labels * dbn.rec.label.pair + ...
		    repmat(dbn.rec.bias{dbn.L+1},N,1));
  h = prob_h > rand(N,dbn.K(end));
  prob_s = logistic(h * dbn.gen.pair{dbn.L} + repmat(dbn.gen.bias{dbn.L},N,1)  );
  s = prob_s > rand(N,dbn.K(end-1));
end
x = s;
for k=dbn.L-1:-1:2
  prob_x = logistic(x * dbn.gen.pair{k} + repmat(dbn.gen.bias{k},N,1));
  x = prob_x > rand(N,dbn.K(k-1));
end
prob_x = logistic(x * dbn.gen.pair{1} + repmat(dbn.gen.bias{1},N,1));
x = prob_x > rand(N,dbn.D);

end
