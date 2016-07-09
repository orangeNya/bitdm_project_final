function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Return the following variables
J = 0;
grad = zeros(size(theta));

%  The cost of a particular choice of theta.
%  We set J to the cost.
%  Compute the partial derivatives and set grad to the partial
%  derivatives of the cost w.r.t. each parameter in theta

expth = exp(- X * theta);
siglog = -log(1 + expth);
signlog = siglog + log(expth);

J = - ( y' * siglog + (1-y)' * signlog ) / m;

grad = X' * (sigmoid(X * theta) - y) ./m; %'

end
