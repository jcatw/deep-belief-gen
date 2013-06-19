function [ dbn ] = dbntrain( x, L, T, B, C, K, G, alpha, lambda )
%DBNTRAIN Train a deep belief network.
%   Train a deep belief network (DBN) via greedy stochastic mini-batch
%   gradient descent.
%   
%   Arguments:
%       x: training data
%       L: number of hidden layers
%       T: number of training iterations per hidden layer
%       B: number of batches per training iteration
%          must be a factor of the number of training instances
%       C: number of parallel chains to use for the monte carlo 
%          approximation to the likelihood gradient.
%       K: vector of length L with the number of hidden variables per layer
%	G: Number of alternating gibbs iterations to take during backfitting
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
assert(L > 0, 'Need at least one hidden layer');
assert(length(K) == L,'Wrong hidden unit length');

[N,D] = size(x);
assert(mod(N,B) == 0,sprintf('Batch size B (currently %d) must be a factor of the number of training instances (%d)',B,N));
Nb = N / B;

% parameters and pre-initialization
dbn.L = L;
dbn.K = K;
dbn.D = D;

dbn.rec.bias = cell(1,L+1);  % work with receptive weights, set generative to result when done
dbn.rec.pair = cell(1,L);

dbn.rec.bias{1} = 0.1 * randn(1,D);
dbn.rec.bias{2} = 0.1 * randn(1,K(1));

dbn.rec.pair{1} = 0.1 * randn(D,K(1));

sample_h = randi(2,C,K(1)) - 1;

% train first layer
for t=1:T
  for b=1:B
    fprintf(1,'Layer 1, iteration %d, batch %d\n',t,b);

    batch_data = x( 1+(b-1)*Nb : b*Nb, : );

    % positive gradient
    g_plus_bias_vis = sum(batch_data, 1);
    plus_p = logistic( batch_data * dbn.rec.pair{1} + repmat(dbn.rec.bias{2},Nb,1) );
    g_plus_bias_hid = sum(plus_p,1);
    g_plus_pair = batch_data' * plus_p;

    % negative gradient
    prob_x = logistic( sample_h * dbn.rec.pair{1}' + repmat(dbn.rec.bias{1},C,1) );
    sample_x = prob_x > rand(C,D);
    prob_h = logistic( sample_x * dbn.rec.pair{1} + repmat(dbn.rec.bias{2},C,1) );
    sample_h = prob_h > rand(C,K(1));

    g_minus_bias_vis = sum(sample_x,1);
    g_minus_bias_hid = sum(prob_h,1);
    g_minus_pair = sample_x' * prob_h;

    % update parameters
    dbn.rec.bias{1} = dbn.rec.bias{1} + alpha*(g_plus_bias_vis/Nb - g_minus_bias_vis/C - lambda*dbn.rec.bias{1});
    dbn.rec.bias{2} = dbn.rec.bias{2} + alpha*(g_plus_bias_hid/Nb - g_minus_bias_hid/C - lambda*dbn.rec.bias{2});
    dbn.rec.pair{1} = dbn.rec.pair{1} + alpha*(g_plus_pair/Nb - g_minus_pair/C - lambda*dbn.rec.pair{1});
  end
end

% train remaining layers
for lyr=2:L
  dbn.rec.pair{lyr} = 0.1 * randn(K(lyr-1),K(lyr));
  dbn.rec.bias{lyr+1} = 0.1 * randn(1,K(lyr));

  sample_h = randi(2,C,K(lyr)) - 1;
  for t=1:T
    for b=1:B
	fprintf(1,'Layer %d, iteration %d, batch %d\n',lyr,t,b);

	% sample current visible layer
	batch_data = x( 1+(b-1)*Nb : b*Nb, : );
	for i=1:lyr-1
	    batch_data = logistic( batch_data * dbn.rec.pair{i} + repmat(dbn.rec.bias{i+1},Nb,1) ) > rand(Nb, dbn.K(i));
	end

	% positive gradient
	plus_p = logistic( batch_data * dbn.rec.pair{lyr} + repmat(dbn.rec.bias{lyr+1},Nb,1) );
	g_plus_bias = sum(plus_p,1);
	g_plus_pair = batch_data' * plus_p;

	% negative gradient
	prob_x = logistic( sample_h * dbn.rec.pair{lyr}' + repmat(dbn.rec.bias{lyr},C,1) );
	sample_x = prob_x > rand(C,K(lyr-1));
	prob_h = logistic( sample_x * dbn.rec.pair{lyr} + repmat(dbn.rec.bias{lyr+1},C,1) );
	sample_h = prob_h > rand(C,K(lyr));

	g_minus_bias = sum(prob_h,1);
	g_minus_pair = sample_x' * prob_h;

	% update parameters
	dbn.rec.pair{lyr} = dbn.rec.pair{lyr} + alpha*(g_plus_pair/Nb - g_minus_pair/C - lambda*dbn.rec.pair{lyr});
	dbn.rec.bias{lyr+1} = dbn.rec.bias{lyr+1} + alpha*(g_plus_bias/Nb - g_minus_bias/C - lambda*dbn.rec.bias{lyr+1});
    end
  end
end

% set generative parameters
dbn.gen.bias = cell(1,L+1);
dbn.gen.pair = cell(1,L);
for i=1:L
    dbn.gen.bias{i} = dbn.rec.bias{i};
    dbn.gen.pair{i} = dbn.rec.pair{i}';
end
dbn.gen.bias(L+1) = dbn.rec.bias(L+1);

% tune via backprop
dbn = dbnbackfit(dbn, x, B, G, alpha, lambda);
end

