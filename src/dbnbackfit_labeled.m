function [ dbn ] = dbnbackfit_labeled( dbn, x, labels, B, G, alpha, lambda )
%DBNBACKFIT_LABELED Tune a trained deep belief network
%   dbnbackfit tunes a trained deep belief network (DBN) via backfitting.  
%   
%   See section 5 of
%   G. E. Hinton, S. Osindero, and Y. W. Teh, 
%   "A fast learning algorithm for deep belief nets,"
%   Neural Computation, 2006.
%
%   Arguments:
%      dbn: A pre-trained deep belief network
%      x: tuning data
%      B: number of batches per training iteration
%         must be a factor of the number of training instances
%      G: number of alternating gibbs samples to take at the top level
%      alpha: learning rate
%      lambda: regularization parameter
%
%   Returns:
%      dbn: a tuned version of the input deep belief net

% James Atwood 6/7/13
% jatwood@cs.umass.edu

[N, D] = size(x);
Nb = N / B;

for b=1:B
  batch_data = x( 1+(b-1)*Nb : b*Nb, : );
  batch_labels = labels( 1+(b-1)*Nb : b*Nb, : );

  % up-pass: pick a state for every hidden variable
  % wakeprobs{i}  = probability of top of layer i
  % wakestates{i} = sampled state of top of layer i
  wakeprobs  = cell(1,dbn.L);
  wakestates = cell(1,dbn.L);
  wakedata = batch_data;
  for i=1:dbn.L
      wakeprobs{i} = logistic( wakedata * dbn.rec.pair{i} + repmat(dbn.rec.bias{i+1},Nb,1) );
      wakestates{i} = double(wakeprobs{i} > rand(Nb,dbn.K(i)));
      wakedata = wakestates{i};
  end

  % gibbs-sample the top-level state
  top_h = wakestates{end};
  for i=1:G
    prob_top_x = logistic( top_h * dbn.gen.pair{end} + repmat(dbn.gen.bias{end-1},Nb,1) ) ;
    top_x = double(prob_top_x > rand(Nb,dbn.K(end-1)));
    %size(dbn.gen.label.bias)
    prob_top_lab = softmax( top_h * dbn.gen.label.pair + repmat(dbn.gen.label.bias,Nb,1) );
    prob_top_h = logistic( top_x * dbn.rec.pair{end} + ...
			   prob_top_lab * dbn.rec.label.pair + ...
			   repmat(dbn.rec.bias{end}  ,Nb,1) );
    top_h = double( prob_top_h > rand(Nb,dbn.K(end  )));
  end

  % down-pass: pick a state for every hidden variable
  % sleepprobs{i} = probability of bottom of layer i
  % sleepstates{i} = sampled state of bottom of layer i
  % no sleep state calculated for layer with visible units
  sleepprobs  = cell(1,dbn.L);
  sleepstates = cell(1,dbn.L);
  sleepprobs{dbn.L} = prob_top_x;
  sleepstates{dbn.L} = top_x;
  for i=dbn.L-1:-1:2
      sleepprobs{i}  = logistic( sleepstates{i+1} * dbn.gen.pair{i} + repmat(dbn.gen.bias{i},Nb,1) );
      sleepstates{i} = double(sleepprobs{i} > rand(Nb,dbn.K(i-1)));
  end
  visprobs = logistic( sleepstates{2} * dbn.gen.pair{1} + repmat(dbn.gen.bias{1},Nb,1) );
  
  % update parameters
  %size(wakestates{1})
  %size(batch_data)
  %size(visprobs)
  %size(wakestates{1}' * (batch_data - visprobs))
  %size(dbn.gen.pair{1})
  dbn.gen.pair{1} = dbn.gen.pair{1} + alpha * (wakestates{1}' * (batch_data - visprobs)) / Nb;
  dbn.gen.bias{1} = dbn.gen.bias{1} + alpha * sum(batch_data - visprobs,1) / Nb;
  for i=2:dbn.L-1
    dbn.gen.pair{i} = dbn.gen.pair{i} + alpha * (wakestates{i}' * (wakestates{i-1}-wakeprobs{i-1})) / Nb;
    dbn.gen.bias{i} = dbn.gen.bias{i} + alpha * sum(wakestates{i-1} - wakeprobs{i-1},1) / Nb;
  end

  %size(wakestates{dbn.L-1})
  %size(wakestates{dbn.L})
  %size(wakestates{dbn.L-1}' * wakestates{dbn.L})
  %size(top_x)
  %size(top_h)
  %size(top_x' * top_h)
  %size(dbn.rec.pair{dbn.L})

  dbn.rec.label.pair = dbn.rec.label.pair + ...
		       alpha*(batch_labels' * wakestates{dbn.L} - prob_top_lab' * top_h);
  dbn.gen.label.pair = dbn.rec.label.pair';
  dbn.gen.label.bias = dbn.gen.label.bias + alpha*sum(batch_labels - prob_top_lab) / Nb;
  
  dbn.rec.pair{dbn.L} = dbn.rec.pair{dbn.L} + alpha*(wakestates{dbn.L-1}' * wakestates{dbn.L} - top_x' * top_h) / Nb;
  dbn.gen.pair{dbn.L} = dbn.rec.pair{dbn.L}';
  dbn.gen.bias{dbn.L} = dbn.gen.bias{dbn.L} + alpha*sum(wakestates{dbn.L-1} - sleepstates{dbn.L},1) / Nb;
  dbn.rec.bias{dbn.L+1} = dbn.rec.bias{dbn.L+1} + alpha*sum(wakestates{dbn.L-1} - top_h,1) / Nb;
  dbn.gen.bias{dbn.L+1} = dbn.rec.bias{dbn.L+1};

  for i=dbn.L-1:-1:2
    %size(sleepstates{i+1})
    %size(sleepprobs{i+1})
    dbn.rec.pair{i} = dbn.rec.pair{i} + alpha*(sleepstates{i}'*(sleepstates{i+1}-sleepprobs{i+1})) / Nb;
    dbn.rec.bias{i+1} = dbn.rec.bias{i+1} + alpha*sum(sleepstates{i+1}-sleepprobs{i+1},1) / Nb;
  end
  dbn.rec.pair{1} = dbn.rec.pair{1} + (visprobs'*(sleepstates{2}-sleepprobs{2})) / Nb;
  dbn.rec.bias{2} = dbn.rec.bias{2} + alpha*sum(sleepstates{2}-sleepprobs{2},1) / Nb;
end
end
