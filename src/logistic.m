function [ logit ] = logistic( t )
%LOGISTIC Evaluate the logistic function

logit = 1 ./ (1 + exp(-t));

end

