function [ samples ] = softmax_sample(probs)
% SOFTMAX_SAMPLE sample from softmax probabilities

[N,c] = size(probs);
unif = rand(N,1);
raw = cumsum(probs,2) > repmat(unif,1,c);

samples = zeros(N,c);
for i=1:N
    for j=1:c
	if raw(i,j)
	  samples(i,j) = 1;
	  break;
	end
    end
end
