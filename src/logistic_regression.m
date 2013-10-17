function [ b, included, dev, stats ] = logistic_regression(X,Y,alpha)
%LOGISTIC_REGRESSION Perform logisitic regression with feature elimination on design matrix X and response Y.
%   Logistic regression with feature elimination.  Assumes that
%   the design matrix X and response Y are binary.  Features are included
%   if the absolute value of their coefficient is greater than or equal to
%   alpha.
%
%   Arguments:
%       X: design matrix
%       Y: response vector
%       alpha: feature inclusion tolerance				  
%
%   Returns:
%      b: coefficients of included features (eliminated features set to 0.)
%      included: boolean vector of feature inclusions			  
%      dev: deviance
%      stats: statistics returned by glmfit

%   James Atwood 8/30/13
%   jatwood@cs.umass.edu

[b, dev, stats] = glmfit(X,Y,'binomial','link','logit','constant','off');
%'b'
%size(b)
%'X'
%size(X)

included = abs(b) >= alpha;

b(~included) = 0.;



