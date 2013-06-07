function [ dbn ] = dbnbackfit( x, dbn, t )
%DBNBACKFIT Tune a trained deep belief network
%   dbnbackfit tunes a trained deep belief network (DBN).  
%   
%   See section 5 of
%   G. E. Hinton, S. Osindero, and Y.-W. Teh, 
%   ?A fast learning algorithm for deep belief nets,? 
%   Neural Computation, 2006.
%
%   Arguments:
%      x: tuning data
%      dbn: A pre-trained deep belief network (as produed by dbntrain)
%      t: number of alternating gibbs samples to take at the top level
%
%   Returns:
%      dbn: a tuned version of the input deep belief network

% James Atwood 6/7/13
% jatwood@cs.umass.edu

% bottom-up pass
lower_state = x;
for i=1:dbn.L
    wakeprob = logistic(lower_state*dbn.rbms{i}.rec.Wp + bn.rbms{i}.rec.Wb);
    wakestates{i} = wakeprob > rand(1,dbn.K{i});
    
    lower_state = wakestates{i};
end

% positive phase statistics
postopstatistics = wakestates{end-1}' * wakestates{end};

% alternating gibbs sampling for t iterations
negtopstates = wakestates{end};
for i=1:t
    negpenprobs = logistic(negtopstates' * dbn.rbms{end}.Wp' + dbn.rbms{end}.Wc);
    negpenstates = negpenprobs > rand(1, dbn.K(end-1));
    
    negtopprobs = logistic(negpenstates*dbn.rbms{end}.Wp + + dbn.rbms{end}.Wb);
    negtopstates = negtopprobs > rand(1, dbn.K(end));
end

% negative phase statistics
negpentopstatistics = negpenstates' * negtopstates;

sleepstates{dbn.L} = negpenstates;

for i=dbn.L-1:-1:1
    sleepprob = logistic(sleepstates{i+1} * dbn.rbms{i}.gen.Wp' + dbn.rbms{i}.Wc);
    sleepstates{i} = sleepprob > rand(1,dbn.K(i-1));
end

% predictions
%  wake
for i=1:dbn.L-1
p_wakestates{i} = logistic(wakestates{i} * rbms{i}.gen.Wp + rbms{i};
end


end