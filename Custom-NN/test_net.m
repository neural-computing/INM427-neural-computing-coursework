function [results,accuracy] = test_net(X_test,y_test,theta1,theta2,predictions)
  
  for n=1:size(X_test,1)
    a1 = [1;X_test(n,:)']; % a1 is a column vector for a sample +1 (for bias)

    % then calculate the hidden layer activations and gradients
    z2 = [1;theta1 * a1]; % 26x1 calculate input potential of hidden layer
    a2 = sigmoid(z2); % 26x1 calculate activation of hiddent layer
    z3 = theta2 * a2;
    a3 = sigmoid(z3);
    [val, idx] = max(a3);
    predictions = [predictions;idx-2];
  end
  results = [predictions y_test];
  % find accuracy 
  tmp = predictions - y_test; %only if elements match will the element be 0
  accuracy = sum(tmp==0) ./ size(X_test,1);
end