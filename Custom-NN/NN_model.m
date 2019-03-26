function [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,time_taken,old_accuracy,overal_error,train_results,test_results] = NN_model(X_train,y_train,X_test,y_test,y_train_explode,n_hidden_nodes,n_classes,weights_scale_factor,learning_rate,epochs,momentum,early_stopping_thresh,early_stopping_resilance,debug)


  %Initialize bias as additional column vector on the front 
  %of the feature matrix, ready for matrix multiplication of weights
  d1=0;
  d2=0;
  old_accuracy = [0];
  old_acc_delta = ones(early_stopping_resilance,1);
  predictions = [];

  %start timer
  t = cputime();

  for iter=1:epochs
    for m=1:size(X_train,1)
      
      %add a input bias
      a1 = [1;X_train(m,:)']; % 10x1  a1 is a column vector for a sample +1 (for bias)
      
      %Next we initialize some weights, calculate the input potential matrix for the hidden layer and 
      % then calculate the hidden layer activations and gradients
      if iter == 1 && m==1
        theta1 = buildWeights(size(a1,1),n_hidden_nodes,weights_scale_factor);
        if debug == true
          disp("weights1 initialized");
        end
      end
      
      z2 = [1;theta1 * a1]; % 26x1 calculate input potential of hidden layer
      a2 = sigmoid(z2); % 26x1 calculate activation of hiddent layer
      g2 = sigmoidGradient(z2); % 26x1 calculate gradients with these activations
      
      if iter == 1 && m==1
        theta2 = buildWeights(size(a2,1),n_classes,weights_scale_factor);
        if debug == true
          disp("weights2 initialized");
        end
      end
      
      z3 = theta2 * a2; % compute input potential of output layer
      a3 = sigmoid(z3); % compute activation of output layer 
      g3 = sigmoidGradient(z3); % compute gradients at output layer - possibly not needed

      %back prop - calculate errors through network
      e3 = a3-y_train_explode(m,:)'; % calculate errors at output layer
      e2 = theta2' * e3 .* g2; % calculate errors at hidden layer 
      overal_error = mean(sum(square(e3))+sum(square(e2))); %calculate overal 
      
      %calculate deltas for each layer
      if iter == 1 && m==1
        d2 = -learning_rate * (e3*a2'); % calculate deltas for theta2
        d1 = -learning_rate * (e2*a1'); % calculate deltas for theta1 
      else
        d2 = -learning_rate * (e3*a2') + momentum * d2; % calculate deltas for theta2
        d1 = -learning_rate * (e2*a1') + momentum * d1; % calculate deltas for theta1 
      end
      
      theta2 = theta2 + d2;
      theta1 = theta1 + d1(2:end,:);
    end
    %impliment early stopping if accuracy improvement less than threshold
    [train_results,train_accuracy] = test_net(X_train,y_train,theta1,theta2,predictions);
    
    %cal accuary delta 
    acc_delta = abs(old_accuracy(iter) - train_accuracy);
    
    if debug == true
      display([acc_delta,old_accuracy(iter),train_accuracy]);
    end
    
    %stop if delta in accuray is less than early_stopping_thresh
    if sum(old_acc_delta(iter:iter+early_stopping_resilance-1)) < early_stopping_thresh
      if debug == true
          disp("accuracy delta less than stopping threshold - stopping training");
          disp([train_accuracy iter]);
      end
      break
    
    end
    
    %update accuracy and log overal error throughout net
    old_accuracy = [old_accuracy;train_accuracy];
    old_acc_delta = [old_acc_delta;acc_delta];
    overal_error = [overal_error;overal_error];
    end
 
  [test_results,test_accuracy] = test_net(X_test,y_test,theta1,theta2,predictions);
  
  %crop off ofset (dur to how we impliment the stopping resilience
  old_acc_delta = old_acc_delta(10:end);
  
  train_confusion = confusionmat(train_results(:,1),train_results(:,2));
  test_confusion = confusionmat(test_results(:,1),test_results(:,2));
  overal_confusion = confusionmat([train_results(:,1);test_results(:,1)],[train_results(:,2);test_results(:,2)]);
  time_taken = cputime() - t;
end
