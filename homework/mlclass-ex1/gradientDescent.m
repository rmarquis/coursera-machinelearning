function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % theta = theta - alpha*dJ/dx((theta*x - y)^2)
    
    % unvectorized
    %update = 0;
    %for i=1:m,
    %    update = update + alpha/m * (theta' * X(i,:)' - y(i)) * X(i,:)';
    %end
    %
    %theta = theta - update;
    
    % vectorized
    theta = theta - alpha * 1/m .* X' * (X * theta - y);

    % H_theta = X * theta;
    % grad = (1/m).* x' * (H_theta - y);
    % theta = theta - alpha .* grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
