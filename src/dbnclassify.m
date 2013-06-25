function [predictions] = dbnclassify(dbn,x,G)

[N,D] = size(x);

probs = x;
% pass data up
for i=1:dbn.L-1
    probs = logistic(probs * dbn.rec.pair{i} + repmat(dbn.rec.bias{i+1},N,1));
    %x = double(probs > rand(N,dbn.K(i)));
end

prob_x = probs;
prob_l = ones(N,dbn.numclasses) / dbn.numclasses;
for i=1:G
    prob_h = logistic(prob_l * dbn.rec.label.pair + ...
		      prob_x * dbn.rec.pair{end} + ...
		      repmat(dbn.rec.bias{end},N,1));
    prob_l = softmax(prob_h * dbn.gen.label.pair + repmat(dbn.gen.label.bias,N,1));
    prob_x = logistic(prob_h * dbn.gen.pair{end} + repmat(dbn.gen.bias{end-1},N,1));
end
%probs = prob_h;

% given the top layer, determine most likely label
%label_probs = softmax(probs * dbn.gen.label.pair + repmat(dbn.gen.label.bias,N,1))
%size(label_probs)
%size(max(label_probs,[],2))
%predictions = double( label_probs == max(label_probs,[],2) );
predictions = double( prob_l == max(prob_l,[],2) );
