pkg load nnet
pkg load parallel
page_output_immediately (1);
format short
more off

clear all;
% Read in Raw csv
%raw = csvread("PhishingData.csv",1); %needed for matlab
raw = csvread("PhishingData.csv");

n_hidden_nodes=10:5:60;
n_classes = 3;
weights_scale_factor = 1;
learning_rate = 0.01;
epochs = 150;
momentum = 0.025;
early_stopping_thresh = 0.001;
early_stopping_resilance = 10;
test_train_split_p = .7;      % proportion of rows to select for training
debug = false;
overal_results=[];
tests = 0;
nproc = 11;

fun = @(x) NN_model(raw,x,n_classes,weights_scale_factor,learning_rate,epochs,momentum,early_stopping_thresh,early_stopping_resilance,test_train_split_p,debug);

 %     NN_model(raw,n_hidden_nodes,n_classes,weights_scale_factor,learning_rate,
  %    epochs,momentum,early_stopping_thresh,early_stopping_resilance,
   %   test_train_split_p,debug)


for learning_rate=0.001:0.05:0.1
  for momentum=0.005:0.025:0.05
    for early_stopping_resilance=5:2:10
      tests = tests+1;
      display(tests.*nproc) #keep count of number of parallel tasks
      [confusion,train_accuracy,test_accuracy,time_taken] = pararrayfun(nproc, fun, n_hidden_nodes,"UniformOutput", false);

     %display(confusion);
     for i=1:size(nproc,2)
      overal_results = [overal_results;[n_hidden_nodes,learning_rate,momentum,early_stopping_resilance,train_accuracy(1,i),test_accuracy(1,i),time_taken(1,i)]];
     endfor
    endfor
  endfor
endfor

overal_results = cell2mat(overal_results);

display(overal_results);