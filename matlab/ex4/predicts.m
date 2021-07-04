function p = predicts(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X = [ones(m, 1), X];
Z1 = X * Theta1';
A2 = sigmoid(Z1);
A2 = [ones(m, 1), A2];
Z2 = A2 * Theta2';
A3 = sigmoid(Z2);
[~, p] = max(A3, [], 2);

% =========================================================================

end
