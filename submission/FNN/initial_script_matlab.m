format short
more off

clear all;
% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab

%Set parameters
n_hidden_nodes=10:4:40;
n_classes = 3; 
weights_scale_factor = 0.2;
learning_rates=0.001:0.015:0.0501;
epochs = 150;
momentums=0.005:0.025:0.05;
early_stopping_threshs=0.001:0.05:0.1;
early_stopping_resilance = 10;
test_train_split_p = .7;      % proportion of rows to select for training
debug = false;
tests = 0;
num_folds = 5;

%shuffle dataset and build k folds as a cell array
raw_folds = kfolds(raw,num_folds);

%initialize some more parameters  
total_tests = size(learning_rates,2) * size(momentums,2) * size(early_stopping_threshs,2) * size(n_hidden_nodes,2);
disp(total_tests);
overal_results = zeros(total_tests,7);
wbm = waitbar(0,sprintf("Training Models (%d)", total_tests));
train_confusions = cell(total_tests,1);
test_confusions = cell(total_tests,1);
overal_confusions = cell(total_tests,1);
train_predictions = cell(total_tests,1);
test_predictions = cell(total_tests,1);
test_confusions_sum = cell(total_tests,1);

accuracy_per_epoch_k = zeros(epochs,num_folds,total_tests);

%how stable is the network? we see that 
for learning_rate=learning_rates
    for momentum=momentums
        for early_stopping_thresh=early_stopping_threshs
            
            for n_hidden_node=n_hidden_nodes
                
                tests = tests+1;
                waitbar((tests/total_tests),wbm,sprintf("Training Models (%d/%d)", [tests,total_tests]));
                
                %build dummy variables
                train_accuracy_k = zeros(num_folds,1);
                test_accuracy_k = zeros(num_folds,1);
                time_taken_k = zeros(num_folds,1);
                train_confusion_k = zeros(3,3,num_folds);
                test_confusion_k = zeros(3,3,num_folds);
                overal_confusion_k = zeros(3,3,num_folds);
                
                %perform kfold
                for k=1:num_folds
                    disp(k)
                    %split for test and train based on 1 vs all on the
                    %kfolds, kth fold becomes test and rest become train
                    [X_train,y_train,X_test,y_test,y_train_explode] = process_data_kfold(raw_folds,k);
                    
                    %run model
                    [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,time_taken,accuracy_per_epoch,error_per_epoch,train_results,test_results]...
                        = NN_model(X_train,y_train,X_test,y_test,y_train_explode,n_hidden_node,n_classes,weights_scale_factor,learning_rate,epochs,...
                        momentum,early_stopping_thresh,early_stopping_resilance,debug);
                    
                    %capture variables
                    train_accuracy_k(k) = train_accuracy; test_accuracy_k(k) = test_accuracy; time_taken_k(k) = time_taken;
                    train_confusion_k(:,:,k) = train_confusion;
                    test_confusion_k(:,:,k) = test_confusion;
                    overal_confusion_k(:,:,k) = overal_confusion;
                    padding = [accuracy_per_epoch ; nan(150-size(accuracy_per_epoch,1),1)];
                    accuracy_per_epoch_k(:,k,tests) = padding(1:150);
                    
                end
                
                display([tests,mean(test_accuracy_k)])
                
                %append to overal results data structure
                overal_results(tests,:) = [n_hidden_node,learning_rate,momentum,early_stopping_thresh,mean(train_accuracy_k),mean(test_accuracy_k),mean(time_taken_k)];
                
                %capture confusion matrix averaged over k folds
                train_confusions{tests} = mean(train_confusion_k,3);
                test_confusions{tests} = mean(test_confusion_k,3);
                test_confusions_sum{tests} = sum(test_confusion_k,3);
                overal_confusions{tests} = mean(overal_confusion_k,3);
                train_predictions{tests} = train_predictions;
                test_predictions{tests} = test_predictions;

            end
        end
    end
end

close(wbm);

%pick best model cadidate using the best accuracy, print its setting and
%its kfold test confusion matrix
[v,ind] = maxk(overal_results(:,6),10);
overal_results(ind,:)
test_confusions{ind}

%test stability
best_nn = overal_results(ind(1),:);

%plot accuracy and error per epoch for each fold of the best model
figure(1)
hold on
for k=1:num_folds
    legend("Location","east")
    x = 1:epochs;
    plot(x,accuracy_per_epoch_k(:,k,ind(1)),'DisplayName', sprintf('Epoch: %d',k))
end

xlabel("Epochs");
ylabel("Training Accuracy");
grid on
grid minor
hold off

disp("saving diagrams")
%saveas(figure(1),"Test-NN-Prob-Dens.png")
saveas(figure(1),"Test-NN-acc.png")
%saveas(figure(3),"Test-NN-err.png")

disp("saving datasets")
writematrix(overal_results,"NN_model_results.csv")
%writematrix(best_distro,"NN_best_distro.csv")
writematrix(overal_results(ind,:),"NN_top_10.csv")

disp("saving  confusion plots")

%plot and save confusion matrix for best model
labels = {'Phishing', 'Non-Phishing', 'Unknown'};
plot_confusion(train_confusions{ind(1)},labels,"Training Set",4)
plot_confusion(test_confusions{ind(1)},labels,"Testing Set",7)
plot_confusion(test_confusions_sum{ind(1)},labels,"5 Kfold set",5)
plot_confusion(overal_confusions{ind(1)},labels,"Overal Set",6)
saveas(figure(4),"best_NN_train_confusion.png")
saveas(figure(7),"best_NN_kfold_average_test_confusion_compare_this.png")
saveas(figure(6),"best_NN_overal_confusion.png")
saveas(figure(5),"best_NN_kfold_sum_test_confusion_compare_this.png")
disp("complete")
