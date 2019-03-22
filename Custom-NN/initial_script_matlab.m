format short
more off

clear all;
% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab
%raw = csvread("PhishingData.csv");

n_hidden_nodes=10:5:60;
n_classes = 3; the carsmall data set. Consider a model that predicts the fuel economy of a car given its engine displacement.
weights_scale_factor = 1;
learning_rates=0.001:0.005:0.1;
epochs = 150;
momentums=0.005:0.025:0.05;
early_stopping_threshs=0.005:0.025:0.05;
early_stopping_resilance = 10;
test_train_split_p = .7;      % proportion of rows to select for training
debug = false;
tests = 0;
nproc = 11;

%build test/train data
[X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p);
  
%initialize some more parameters  
total_tests = size(learning_rates,2) * size(momentums,2) * size(early_stopping_threshs,2) * size(n_hidden_nodes,2);
overal_results = zeros(total_tests,7);
wbm = waitbar(0,sprintf("Training Models (%f)", total_tests));
confusions = cell(total_tests,3,3);

for learning_rate=learning_rates
    for momentum=momentums
        for early_stopping_thresh=early_stopping_threshs
            
            %[confusion,train_accuracy,test_accuracy,time_taken] = pararrayfun(nproc, fun, n_hidden_nodes,"UniformOutput", false);
            for n_hidden_node=n_hidden_nodes
                tests = tests+1;
                waitbar((tests/total_tests),wbm,"Training Models");
                [confusion,train_accuracy,test_accuracy,time_taken] = NN_model(X_train,y_train,X_test,y_test,y_train_explode,n_hidden_node,n_classes,weights_scale_factor,learning_rate,epochs,momentum,early_stopping_thresh,early_stopping_resilance,debug);
                display([tests,test_accuracy])
                overal_results(tests,:) = [n_hidden_node,learning_rate,momentum,early_stopping_resilance,train_accuracy,test_accuracy,time_taken];
                confusions{tests} = confusion;
            end
        end
    end
end

close(wbm);

%pick best test
[v,i] = max(overal_results(:,6));
display(overal_results(i,:));
display(confusions{i});


