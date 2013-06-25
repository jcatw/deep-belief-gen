function [predictions] = dbnclassify_simple(dbn,x)

[N,D] = size(x);

probs = x;
% pass data up
for i=1:dbn.L
    probs = logistic(probs * dbn.rec.pair{i} + repmat(dbn.rec.bias{i+1},N,1));
    %x = double(probs > rand(N,dbn.K(i)));
end

% given the top layer, determine most likely label
label_probs = softmax(probs * dbn.gen.label.pair + repmat(dbn.gen.label.bias,N,1));
%size(label_probs)
%size(max(label_probs,[],2))
predictions = double( label_probs == max(label_probs,[],2) );
