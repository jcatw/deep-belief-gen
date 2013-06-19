function [ samples ] = dbnsample_labeled( dbn, label, N, G, burn_in )
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

% init to random binary visible state and sample to penultimate layer
s = randi(2,1,dbn.D) - 1;  
for k=1:dbn.L-1
  p = logistic(s * dbn.rec.pair{k} + dbn.rec.bias{k+1});
  s = p > rand(1,dbn.K(k));
end

% burn-in the top level
for b=1:burn_in
  for j=1:G
    prob_h = logistic(s * dbn.rec.pair{dbn.L} + ...
		      label * dbn.rec.label.pair + ...
		      dbn.rec.bias{dbn.L+1});
    h = prob_h > rand(1,dbn.K(end));
    prob_s = logistic(h * dbn.gen.pair{dbn.L} + dbn.gen.bias{dbn.L}  );
    s = prob_s > rand(1,dbn.K(end-1));
  end
end

% perform alternating gibbs sampling with a down pass every G iterations
for i=1:N
  for j=1:G
    prob_h = logistic(s * dbn.rec.pair{dbn.L} + ...
		      label * dbn.rec.label.pair + ...
		      dbn.rec.bias{dbn.L+1});
    h = prob_h > rand(1,dbn.K(end));
    prob_s = logistic(h * dbn.gen.pair{dbn.L} + dbn.gen.bias{dbn.L}  );
    s = prob_s > rand(1,dbn.K(end-1));
  end
  x = s;
  for k=dbn.L-1:-1:2
    prob_x = logistic(x * dbn.gen.pair{k} + dbn.gen.bias{k});
    x = prob_x > rand(1,dbn.K(k-1));
  end
  prob_x = logistic(x * dbn.gen.pair{1} + dbn.gen.bias{1});
  x = prob_x > rand(1,dbn.D);
  samples(i,:) = x;
end
end

