format short
more off
clear all;

% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab

%parameters
test_train_split_p = .7;      % proportion of rows to select for training
n_trees = 10:20:150;
min_leaf_sizes = 1:5;
in_bag_fractions=0.5:0.1:1;

%build test/train data
[X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p);
 
%initialize some more parameters  
total_tests = size(n_trees,2) * size(min_leaf_sizes,2) * size(in_bag_fractions,2);
overal_results = zeros(total_tests,5);
wbm = waitbar(0,sprintf("Training Models (%f)", total_tests));
confusions = cell(total_tests,3,3);
tests = 0;

for n_tree=n_trees
    for min_leaf_size=min_leaf_sizes
        for in_bag_fraction=in_bag_fractions
            %interate counter
            tests = tests+1;
            
            %Build model
            Mdl = TreeBagger(n_tree,X_train,y_train,'OOBPrediction','On','Method','classification','MinLeafSize',min_leaf_size,'InBagFraction',in_bag_fraction);

            %Predict
            train_predictions = str2double(Mdl.predict(X_train));
            test_predictions = str2double(Mdl.predict(X_test));

            %[str2double(predictions) y_test]
            tmp_train = train_predictions - y_train;
            tmp_test = test_predictions - y_test; %only if elements match will the element be 0

            %Build confusion matrix
            confusions{tests} = confusionmat(test_predictions,y_test);

            %Build Accuracy
            train_accuracy = sum(tmp_train==0) ./ size(X_train,1);
            test_accuracy = sum(tmp_test==0) ./ size(X_test,1);
            
            %append results
            overal_results(tests,:) = [n_tree,min_leaf_size,in_bag_fraction,train_accuracy,test_accuracy];
            
            %update waiting back
            waitbar((tests/total_tests),wbm,"Training Models");

        end
    end
end

display(overal_results);
close(wbm);

%pick best test
[v,i] = max(overal_results(:,5));
display(overal_results(i,:));
display(confusions{i});