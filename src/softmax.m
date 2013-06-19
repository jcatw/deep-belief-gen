function [ probs ] = softmax( x )
%SOFTMAX compute the softmax probabilities of the rows of x

[N,c] = size(x);
Z = sum(exp(x),2);
probs = exp(x) ./ repmat(Z,1,c);
