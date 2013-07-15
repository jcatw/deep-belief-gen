function [dbn] = dbngradientstep(dbn, lyr, batch_data, sample_h, C, alpha, lambda, bottom = 0, batch_labels = [])

%   James Atwood 7/15/13
%   jatwood@cs.umass.edu
	 
K = dbn.K;
D = dbn.D;
Nb = dbn.Nb;

if(lyr == 1)
  vislen = D;
else
  vislen = K(lyr-1);
end
hidlen = K(lyr);

% positive gradient
if(bottom)
  g_plus_bias_vis = sum(batch_data, 1);
end
if(~isempty(batch_labels))
  g_plus_label_bias = sum(batch_labels,1);
end

plus_p_terms = batch_data * dbn.rec.pair{lyr} + repmat(dbn.rec.bias{lyr+1},Nb,1);
if(~isempty(batch_labels))
  plus_p_terms = plus_p_terms + batch_labels * dbn.rec.label.pair;
end
plus_p = logistic( plus_p_terms );

g_plus_bias_hid = sum(plus_p,1);
g_plus_pair = batch_data' * plus_p;
if(~isempty(batch_labels))
  g_plus_label_pair = batch_labels' * plus_p;
end

% negative gradient
prob_x = logistic( sample_h * dbn.rec.pair{lyr}' + repmat(dbn.rec.bias{lyr},C,1) );
sample_x = prob_x > sparse(rand(C,vislen));

if(~isempty(batch_labels))
  prob_lab = softmax( sample_h * dbn.rec.label.pair' + repmat(dbn.rec.label.bias,C,1) );
  sample_lab = softmax_sample(prob_lab);
end

prob_h_term = sample_x * dbn.rec.pair{lyr} + repmat(dbn.rec.bias{lyr+1},C,1);
if(~isempty(batch_labels))
  prob_h_term = prob_h_term + sample_lab * dbn.rec.label.pair;
end
prob_h = logistic( prob_h_term );
sample_h = prob_h > sparse(rand(C,hidlen));

if(bottom)
  g_minus_bias_vis = sum(sample_x,1);
end
g_minus_bias_hid = sum(prob_h,1);
g_minus_pair = sample_x' * prob_h;
if(~isempty(batch_labels))
  g_minus_label_pair = prob_lab' * prob_h;
  g_minus_label_bias = sum(prob_lab,1);
end


% update parameters
if(bottom)
  dbn.rec.bias{lyr} = dbn.rec.bias{lyr} + alpha*(g_plus_bias_vis/Nb - g_minus_bias_vis/C - lambda*dbn.rec.bias{lyr});
end
if(~isempty(batch_labels))
  dbn.rec.label.bias = dbn.rec.label.bias + ...
		       alpha*(g_plus_label_bias/Nb - g_minus_label_bias/C - lambda*dbn.rec.label.bias);
  dbn.rec.label.pair = dbn.rec.label.pair + ...
		       alpha*(g_plus_label_pair/Nb - g_minus_label_pair/C - lambda*dbn.rec.label.pair);
end
dbn.rec.bias{lyr+1} = dbn.rec.bias{lyr+1} + alpha*(g_plus_bias_hid/Nb - g_minus_bias_hid/C - lambda*dbn.rec.bias{lyr+1});
dbn.rec.pair{lyr} = dbn.rec.pair{lyr} + alpha*(g_plus_pair/Nb - g_minus_pair/C - lambda*dbn.rec.pair{lyr});	
