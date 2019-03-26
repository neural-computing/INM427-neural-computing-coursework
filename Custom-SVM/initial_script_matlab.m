format short
more off
clear all;

%fix seed (during testing)
%rng(123)

% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab

%parameters
test_train_split_p = .7;      % proportion of rows to select for training
kernels = ["linear","rbf"];%,"polynomial"];
box_constraints = 0.05:0.3:1;
kernel_scales = 0.1:0.3:1;
shrinkage_periods = 1:3:10;

% dataset parameters
predictorNames = {'SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor',...
    'web_traffic','URL_Length','age_of_domain','having_IP_Address'};
responseName = 'Result';
classNames = {'Phishing Event','Non-Phishing Event','Unknown'}; % Specify class order

%build test/train data
[X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p);
[X,y,X_0,y_0,y_explode] = process_data(raw,1);

%convert y variables to strings
[y_str,y_train_str,y_test_str] = convert_targets_to_strings(y,y_train,y_test);

%initialize some more parameters  
total_tests = size(kernels,2) * size(box_constraints,2) * size(kernel_scales,2) * size(shrinkage_periods,2);
overal_results = zeros(total_tests,7);
wbm = waitbar(0,sprintf("Training Models (%d)", total_tests));
train_confusions = cell(total_tests,1);
test_confusions = cell(total_tests,1);
overal_confusions = cell(total_tests,1);
train_predictions = cell(total_tests,1);
test_predictions = cell(total_tests,1);
cross_fold_errors = cell(total_tests,1);
validationPredictions = cell(total_tests,1);
kfold_confusions = cell(total_tests,1);

tests = 0;

%Run through tests
for kernel=kernels
    for box_constraint=box_constraints
        for kernel_scale=kernel_scales
            for shrinkage_period=shrinkage_periods
                %initiate counter
                tests = tests+1;
                
                %pass variables to SVM wrapper
                [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,train_predictions_str,...
                    test_predictions_str,cross_fold_error,validationPrediction,kfold_confusion]...
                    = SVM_wrapper(X,y,y_str,X_train,y_train,X_test,y_test,y_train_str,y_test_str,kernel,box_constraint,...
                    kernel_scale,shrinkage_period,responseName,predictorNames,classNames);
                
                %append results
                train_confusions{tests} = train_confusion;
                test_confusions{tests} = test_confusion;
                overal_confusions{tests} = overal_confusion;
                train_predictions{tests} = train_predictions_str;
                test_predictions{tests} = test_predictions_str;
                cross_fold_errors{tests} = cross_fold_error;
                validationPredictions{tests} = validationPrediction;
                kfold_confusions{tests} = kfold_confusion;

                %handle the fact that matrix cannot have mixed data types
                if kernel=="linear"
                    overal_results(tests,:) = [box_constraint,kernel_scale,shrinkage_period,1,train_accuracy,test_accuracy,cross_fold_error];
                else
                    overal_results(tests,:) = [box_constraint,kernel_scale,shrinkage_period,2,train_accuracy,test_accuracy,cross_fold_error];
                end
                %update waiting back
                waitbar((tests/total_tests),wbm,sprintf("Training Models (%d/%d)", [tests,total_tests]));

            end
        end
    end
end
 
disp(overal_results);
close(wbm);

%pick best test - use mink since this is error not accuracy
[v,ind] = mink(overal_results(:,7),10);
overal_results(ind,:)
test_confusions{ind}

%test stability
best_svm = overal_results(ind(1),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test stability of top model - when the seed for matlabs random number
%generator is fixed, we get the same output. Otherwise we see variation.
best_distro = zeros(20,3);

for i=1:20

    %pass variables to SVM wrapper
    if best_svm(4)==1
        [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,train_predictions_str,...
            test_predictions_str,cross_fold_error,validationPrediction,kfold_confusion]...
            = SVM_wrapper(X,y,y_str,X_train,y_train,X_test,y_test,y_train_str,y_test_str,"linear",best_svm(1),best_svm(2),best_svm(3),...
            responseName,predictorNames,classNames);
    else
        [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,train_predictions_str,...
            test_predictions_str,cross_fold_error,validationPrediction,kfold_confusion]...
            = SVM_wrapper(X,y,y_str,X_train,y_train,X_test,y_test,y_train_str,y_test_str,"rbf",best_svm(1),best_svm(2),best_svm(3),...
            responseName,predictorNames,classNames);
    end
    best_distro(i,:) = [i,train_accuracy,test_accuracy];
    disp([i test_accuracy])
end
disp(best_distro);

%these are randomly sampled events so we can model as a normal distribution
[m,st] = normfit(best_distro(:,2:end));

%and plot to see variation within the distributions
figure(7)
hold on
x = [0.8:0.001:1];
y1 = normpdf(x,m(1),st(1));
y2 = normpdf(x,m(2),st(2));
mean1 = line([m(1), m(1)], [0 60], 'Color', [0, .6, 0], 'LineWidth', 0.1);
mean2 = line([m(2), m(2)], [0 60], 'Color', [0, .6, 0], 'LineWidth', 0.1);
plot(x,y1)
plot(x,y2)
plot(x,mean1)
plot(x,mean2)
legend('Train Distribution','Test Distribution')
xlabel("Accuracy");
ylabel("Probability Density for 'Best' Model Accuracy");
grid on
grid minor
hold off

disp("saving datasets")
writematrix(overal_results,"SVM_model_results.csv")
writematrix(best_distro,"SVM_best_distro.csv")
writematrix(overal_results(ind,:),"SVM_top_10.csv")
saveas(figure(7),"test.png")

%plot and save confusion matrix for best model
disp("saving confusion plots")
labels = {'Phishing', 'Non-Phishing', 'Unknown'};
plot_confusion(train_confusions{ind(1)},labels,"Training Set",8)
plot_confusion(test_confusions{ind(1)},labels,"Testing Set",9)
plot_confusion(overal_confusions{ind(1)},labels,"Overal Set",10)
plot_confusion(kfold_confusions{ind(1)},labels,"5 Kfold Set",11)
saveas(figure(8),"best_SVM_train_confusion_non_cross_val.png")
saveas(figure(9),"best_SVM_test_confusion_non_cross_val.png")
saveas(figure(10),"best_SVM_overal_confusion_non_cross_val.png")
saveas(figure(11),"best_SVM_kfold_test_confusion_compare_this.png")
disp("complete")