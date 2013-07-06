function [ logit ] = logistic( t )
%LOGISTIC Evaluate the logistic function

logit = sparse(1 ./ (1 + exp(-t)));

end

