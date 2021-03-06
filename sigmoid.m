function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% Return the following variables correctly 
g = zeros(size(z));

%  Compute the sigmoid of each value of z (z can be a matrix,
%  vector or scalar).

g = 1.0 ./ (1.0 + exp(-z));

end
