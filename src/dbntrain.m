function [ dbn ] = dbntrain( x, L, T, Tb, B, C, K, G, alpha, lambda, labels=[] )
%DBNTRAIN Train a deep belief network.
%   Train a deep belief network (DBN) via greedy stochastic mini-batch
%   gradient descent.
%   
%   Arguments:
%       x: training data
%       labels: training labels
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
[N,D] = size(x);

[dummy, lablen] = size(labels);

dbn = dbninit(N, B, K, D, L, lablen);

Nb = dbn.Nb;

sample_h = sparse(randi(2,C,K(1)) - 1);

% train first layer
for t=1:T
  for b=1:B
    fprintf(1,'Pretraining: layer 1, iteration %d, batch %d\n',t,b);

    batch_data = x( 1+(b-1)*Nb : b*Nb, : );
    
    dbn = dbngradientstep(dbn,
			  1,
			  batch_data,
			  sample_h,
			  C,
			  alpha,
			  lambda,
			  1);

  end
end

% train all remaining layers except top
layer_x = x;
for lyr=2:L-1
  % at each layer, initialize new hidden values and upsample observed data
  sample_h = sparse(randi(2,C,K(lyr)) - 1);
  layer_x = logistic( layer_x * dbn.rec.pair{lyr-1} + repmat(dbn.rec.bias{lyr},N,1) ) > sparse(rand(N, dbn.K(lyr-1)));
  for t=1:T
    for b=1:B
	fprintf(1,'Pretraining: layer %d, iteration %d, batch %d\n',lyr,t,b);
	
	% sample current visible layer
	batch_data = layer_x( 1+(b-1)*Nb : b*Nb, : );

	dbn = dbngradientstep(dbn,
			      lyr,
			      batch_data,
			      sample_h,
			      C,
			      alpha,
			      lambda,
			      0);

    end
  end
end

% train top layer
lyr = L;
% re-init h and upsample visible data
sample_h = sparse(randi(2,C,K(lyr)) - 1);
layer_x = logistic( layer_x * dbn.rec.pair{lyr-1} + repmat(dbn.rec.bias{lyr},N,1) ) > sparse(rand(N, dbn.K(lyr-1)));
for t=1:T
  for b=1:B
    fprintf(1,'Pretraining: layer %d, iteration %d, batch %d\n',lyr,t,b);

    % sample current visible layer
    batch_data = layer_x( 1+(b-1)*Nb : b*Nb, : );
    if(~isempty(labels))
      batch_labels = labels( 1+(b-1)*Nb : b*Nb, : );
      dbn = dbngradientstep(dbn,
			    lyr,
			    batch_data,
			    sample_h,
			    C,
			    alpha,
			    lambda,
			    1,
			    batch_labels);
    else
      dbn = dbngradientstep(dbn,
			    lyr,
			    batch_data,
			    sample_h,
			    C,
			    alpha,
			    lambda,
			    1);
    end
  end
end

%set generative parameters
dbn = dbninitgen(dbn);

% tune via backprop
dbn = dbnbackfit(dbn, x, Tb, B, G, alpha, lambda, labels);
end

