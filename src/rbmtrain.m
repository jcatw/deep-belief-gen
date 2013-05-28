function [ rbm ] = rbmtrain( x, T, B, C, K, alpha, lambda )
%RBMTRAIN Train a restricted Boltzmann machine (RBM)
%   Train a restricted Boltzmann machine via stochastic mini-batch
%   gradient descent.
%   
%   Arguments:
%       x: training data
%       T: number of training iterations
%       B: number of batches per training iteration
%       C: number of parallel chains to use for the monte carlo 
%          approximation to the likelihood gradient.
%       K: number of hidden variables
%       alpha: learning rate
%       lambda: regularization parameter
%
%   Returns:
%   Structure rbm
%       rbm.Wb: hidden unit bias potential vector
%       rbm.Wc: visible unit bias potential vector
%       rbm.Wp: pairwise potential matrix
%
%   Example:
%   rbm = rbmtrain(MNISTX, 50, 100, 100, 400, 0.1, 0.0001);
   
%   James Atwood 5/27/13
%   jatwood@cs.umass.edu

[N,D] = size(x);
Nb = N / B;

sample_h = randi(2,C,K) - 1;

W_b = 0.1*randn(K,1); %column
W_c = 0.1*randn(1,D);
W_p = 0.1*randn(D,K);

g_plus_W_b = zeros(K,1); %column
g_plus_W_c = zeros(1,D);
g_plus_W_p = zeros(D,K);

g_minus_W_b = zeros(K,1); %column 
g_minus_W_c = zeros(1,D);
g_minus_W_p = zeros(D,K);

for t=1:T
    t
    for b=1:B
        %'batch x'
        batch_x = x( 1+(b-1)*Nb : b*Nb, : );
        %size(batch_x)
        
        % positive gradient costribution
        g_plus_W_c = sum( batch_x, 1);
        
        %'W_b'
        %size(W_b)
        
        %'repmat W_b'
        %size(repmat(W_b,1,Nb))
        
        %'batch_x * Wp'
        %size(batch_x * W_p)
        
        plus_p = exp(repmat(W_b,1,Nb)' + (batch_x * W_p)) ./ (1 + exp(repmat(W_b,1,Nb)' + (batch_x * W_p)));
        
        %'plus_p'
        %size(plus_p)
        
        g_plus_W_b = sum(plus_p,1)';
        
        g_plus_W_p = batch_x' * plus_p;
        
        % negative gradient contribution
        
        %'sample_h'
        %size(sample_h)
        
        %'W_c'
        %size(W_c)
        
        %'repmat W_c'
        %size(repmat(W_c,C,1))
        
        %'W_p'
        %size(W_p)
        
        %'sample_h * W_pt'
        %size(sample_h * W_p')
        
        prob_x = exp(repmat(W_c,C,1) + (sample_h * W_p')) ./ (1 + exp(repmat(W_c,C,1) + (sample_h * W_p')));
        %sample_x = repmat(prob_x,1,C) < rand(C,D);
        sample_x = prob_x > rand(C,D);
        
        %'sample_x * W_p'
        %size(sample_x * W_p)
        
        %'repmat W_b t'
        %size(repmat(W_b,1,C)')
        
        prob_h = exp(repmat(W_b,1,C)' + (sample_x * W_p)) ./ (1 + exp(repmat(W_b,1,C)' + (sample_x * W_p)));
        sample_h = prob_h > rand(C,K);
        
        g_minus_W_c = sum(sample_x,1);
        
        %'prob_h'
        %size(prob_h)
        
        %'sample_h'
        %size(sample_h)
        
        minus_p = prob_h;
        
        %g_minus_p = sum(minus_p,1);
        g_minus_p = minus_p;

        %'sample_x t'
        %size(sample_x')
        
        %'minus_p'
        %size(minus_p)
        
        %'g_minus_p'
        %size(g_minus_p)
        
        g_minus_W_p = sample_x' * g_minus_p;
        
        %'g_minus_W_p'
        %size(g_minus_W_p)
        
        % gradient step
        W_c = W_c + alpha*(g_plus_W_c / Nb - g_minus_W_c / C - lambda * W_c);
        
        %'W_b'
        %size(W_b)
        %'g_plus_W_b'
        %size(g_plus_W_b)
        %'g_minus_W_b'
        %size(g_minus_W_b)
        
        W_b = W_b + alpha*(g_plus_W_b / Nb - g_minus_W_b / C - lambda * W_b);
        
        W_p = W_p + alpha *(g_plus_W_p / Nb - g_minus_W_p / C - lambda * W_p);
    end
end

rbm.Wb = W_b;
rbm.Wc = W_c;
rbm.Wp = W_p;


end

