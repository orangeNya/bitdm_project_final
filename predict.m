function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% Return the following variables correctly
p = zeros(m, 1);

%  Mmake predictions using our learned logistic regression parameters. 
%  Set p to a vector of 0's and 1's

p = round(sigmoid(X * theta));

end
