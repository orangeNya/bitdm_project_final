function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Return the following variables
J = 0;
grad = zeros(size(theta));

expth = exp(- X * theta);
siglog = -log(1 + expth);
signlog = siglog + log(expth);
J = - ( y' * siglog + (1-y)' * signlog ) / m + theta' * theta * lambda / (2 * m) - ...
theta(1) * theta(1) * lambda / (2 * m);

grad = X' * (sigmoid(X * theta) - y) ./m + theta .* (lambda / m);
grad(1) = grad(1) - theta(1) * (lambda / m);

end
