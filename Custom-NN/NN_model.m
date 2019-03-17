function [confusion,train_accuracy,test_accuracy,time_taken] = NN_model(raw,n_hidden_nodes,n_classes,weights_scale_factor,learning_rate,epochs,momentum,early_stopping_thresh,early_stopping_resilance,test_train_split_p,debug)
  %split data into test and train sectionsq
  N = size(raw,1); % total number of rows 
  tf = false(N,1);    % create logical index vector
  tf(1:round(test_train_split_p*N)) = true;     
  tf = tf(randperm(N));   % randomise order
  dataTraining = raw(tf,:); 
  dataTesting = raw(~tf,:);

  %split raw features from targets
  X_train = dataTraining(:,1:9);
  y_train = dataTraining(:,10);
  %split raw features from targets on test data
  X_test = dataTesting(:,1:9);
  y_test = dataTesting(:,10);

  %one hot encode target variables for use within the matlab neural network
  %toolset, we add 2 to the target variables as well so that our target
  %classes are 1 2 and 3 rather than -1 0 and 1.
  y_train_explode = full(ind2vec(y_train'+2)');

  %Initialize bias as additional column vector on the front 
  %of the feature matrix, ready for matrix multiplication of weights
  d1=0;
  d2=0;
  old_accuracy = [0];
  old_acc_delta = ones(early_stopping_resilance,1);
  null = 0;
  predictions = [];

  %start timer
  t = cputime();
  wb = waitbar(0,"Training Network");

  for iter=1:epochs
    waitbar(iter/epochs,wb,"Training Network");
    for m=1:size(X_train,1)
      
      #%add a input bias
      a1 = [1;X_train(m,:)']; # 10x1  a1 is a column vector for a sample +1 (for bias)
      
      %Next we initialize some weights, calculate the input potential matrix for the hidden layer and 
      % then calculate the hidden layer activations and gradients
      if iter == 1 && m==1
        theta1 = buildWeights(size(a1,1),n_hidden_nodes,weights_scale_factor);
        if debug == true
          disp("weights1 initialized");
        endif
      endif
      
      z2 = [1;theta1 * a1]; % 26x1 calculate input potential of hidden layer
      a2 = sigmoid(z2); % 26x1 calculate activation of hiddent layer
      g2 = sigmoidGradient(z2); % 26x1 calculate gradients with these activations
      
      if iter == 1 && m==1
        theta2 = buildWeights(size(a2,1),n_classes,weights_scale_factor);
        if debug == true
          disp("weights2 initialized");
        endif
      endif
      
      z3 = theta2 * a2; % compute input potential of output layer
      a3 = sigmoid(z3); % compute activation of output layer 
      g3 = sigmoidGradient(z3); % compute gradients at output layer - possibly not needed

      %back prop - calculate errors through network
      e3 = a3-y_train_explode(m,:)'; % calculate errors at output layer
      e2 = theta2' * e3 .* g2; % calculate errors at hidden layer 

      %calculate deltas for each layer
      if iter == 1 && m==1
        d2 = -learning_rate * (e3*a2'); % calculate deltas for theta2
        d1 = -learning_rate * (e2*a1'); % calculate deltas for theta1 
      else
        d2 = -learning_rate * (e3*a2') + momentum * d2; % calculate deltas for theta2
        d1 = -learning_rate * (e2*a1') + momentum * d1; % calculate deltas for theta1 
      endif
      
      theta2 = theta2 + d2;
      theta1 = theta1 + d1(2:end,:);
    endfor
    %impliment early stopping if accuracy improvement less than threshold
    [train_results,train_accuracy] = test_net(X_train,y_train,theta1,theta2,predictions);
    
    #cal accuary delta 
    acc_delta = abs(old_accuracy(iter) - train_accuracy);
    
    if debug == true
      display([acc_delta,old_accuracy(iter),train_accuracy]);
    endif
    
    #stop if delta in accuray is less than early_stopping_thresh
    if sum(old_acc_delta(iter:iter+early_stopping_resilance-1)) < early_stopping_thresh
      %null = null+1;
      %if null==early_stopping_resilance
      display("accuracy delta less than stopping threshold - stopping training");
      display(train_accuracy);
      display(iter);
      waitbar(1,wb,"Training C");
      break
      %endif
    endif
    
    #update accuracy
    old_accuracy = [old_accuracy;train_accuracy];
    old_acc_delta = [old_acc_delta;acc_delta];
  endfor
  disp("finished training");
  

  [test_results,test_accuracy] = test_net(X_test,y_test,theta1,theta2,predictions);

  %show results in a confusion matrix
  %results matrix has to be one hot encoded to work.
  [confusion,rate] = confmat(full(ind2vec(test_results(:,1)+2)'),full(ind2vec(test_results(:,2)+2)'));
  %time taken
  time_taken = cputime() - t;
  close(wb);
endfunction
