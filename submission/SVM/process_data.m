function [X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p)
  %split data into test and train sectionsq
  N = size(raw,1); %total number of rows 
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
  
  
  