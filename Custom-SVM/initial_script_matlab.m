format short
more off
clear all;

%fix seed (during testing)
%rng(123)

% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab

%parameters
test_train_split_p = .7;      % proportion of rows to select for training
box_constraints = 0.1:0.1:1;
kernel_scales = 0.1:0.1:1;
shrinkage_periods = 1:3:10;

% SVM parameters
predictorNames = {'SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor',...
    'web_traffic','URL_Length','age_of_domain','having_IP_Address'};
responseName = 'Result';
classNames = {'Phishing Event','Non-Phishing Event','Unknown'}; % Specify class order

%build test/train data
[X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p);

y_train_str = strrep(cellstr(num2str(y_train)),"-1","Unknown");
y_test_str =  strrep(cellstr(num2str(y_test)),"-1","Unknown");
y_train_str = strrep(y_train_str," 0","Non-Phishing Event");
y_test_str =   strrep(y_test_str," 0","Non-Phishing Event");
y_train_str = strrep(y_train_str," 1","Phishing Event");
y_test_str =   strrep(y_test_str," 1","Phishing Event");

%initialize some more parameters  
total_tests = size(box_constraints,2) * size(kernel_scales,2) * size(shrinkage_periods,2);
overal_results = zeros(total_tests,5);
wbm = waitbar(0,sprintf("Training Models (%f)", total_tests));
confusions = cell(total_tests,3,3);
tests = 0;

%Run through tests
for box_constraint=box_constraints
    for kernel_scale=kernel_scales
        for shrinkage_period=shrinkage_periods
            %initiate counter
            tests = tests+1;
            
             t = templateSVM('Standardize',false,'SaveSupportVectors',true,'BoxConstraint',box_constraint,...
                 'KernelFunction','linear','KernelScale',kernel_scale,'ShrinkagePeriod',shrinkage_period);

             Mdl = fitcecoc(X_train,y_train_str,'Learners',t,'ResponseName',responseName,...
                 'PredictorNames',predictorNames,'ClassNames',classNames);

             %Methods for cross validation
             %CVMdl = crossval(Mdl,'KFold',5);
             %kfoldLoss(CVMdl)

             %Predict
             train_predictions =Mdl.predict(X_train);
             test_predictions = Mdl.predict(X_test);

             %convert test predictions back to intergers
             test_predictions_str =  strrep(test_predictions,"Unknown","-1");
             test_predictions_str =   strrep(test_predictions_str,"Non-Phishing Event"," 0");
             test_predictions_str =   str2num(cell2mat(strrep(test_predictions_str,"Phishing Event"," 1")));

             %Build confusion matrix
             confusions{tests} = confusionmat(test_predictions_str,y_test);

             %Build Accuracy
             train_accuracy = sum(train_predictions == y_train_str) ./ size(y_train,1);
             test_accuracy = sum(test_predictions == y_test_str) ./ size(y_test,1)
            
            %append results
            overal_results(tests,:) = [box_constraint,kernel_scale,shrinkage_period,train_accuracy,test_accuracy];
            
            %update waiting back
            waitbar((tests/total_tests),wbm,"Training Models");

        end
    end
end
 
disp(overal_results);
close(wbm);

%pick best test
[v,ind] = maxk(overal_results(:,5),10);
overal_results(ind,:)
confusions{ind}

%test stability
best_svm = overal_results(ind(1),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test stability of top model - when the seed for matlabs random number
%generator is fixed, we get the same output. Otherwise we see variation.
best_distro = zeros(20,3);

for i=1:20
    t = templateSVM('Standardize',false,'SaveSupportVectors',true,'BoxConstraint',best_svm(1),...
     'KernelFunction','linear','KernelScale',best_svm(2),'ShrinkagePeriod',best_svm(3));

     Mdl = fitcecoc(X_train,y_train_str,'Learners',t,'ResponseName',responseName,...
         'PredictorNames',predictorNames,'ClassNames',classNames);

     %Methods for cross validation
     %CVMdl = crossval(Mdl,'KFold',5);
     %kfoldLoss(CVMdl)

     %Predict
     train_predictions =Mdl.predict(X_train);
     test_predictions = Mdl.predict(X_test);

     %convert test predictions back to intergers
     test_predictions_str =  strrep(test_predictions,"Unknown","-1");
     test_predictions_str =   strrep(test_predictions_str,"Non-Phishing Event"," 0");
     test_predictions_str =   str2num(cell2mat(strrep(test_predictions_str,"Phishing Event"," 1")));

     %Build confusion matrix
     confusions{tests} = confusionmat(test_predictions_str,y_test);

     %Build Accuracy
     train_accuracy = sum(train_predictions == y_train_str) ./ size(y_train,1);
     test_accuracy = sum(test_predictions == y_test_str) ./ size(y_test,1);
     best_distro(i,:) = [i,train_accuracy,test_accuracy];
     disp([i test_accuracy])
end
disp(best_distro);

%these are randomly sampled events so we can model as a normal distribution
[m,st] = normfit(best_distro(:,2:end));

%and plot to see variation within the distributions
figure(3)
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
hold off

writematrix(overal_results,"SVM_model_resullts.csv")
writematrix(best_distro,"SVM_best_distro.csv")
saveas(figure(3),"test.png")