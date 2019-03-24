function [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,train_predictions_str,test_predictions_str] = ...
    SVM_wrapper(X_train,y_train,X_test,y_test,y_train_str,y_test_str,kernel,box_constraint,kernel_scale,...
    shrinkage_period,responseName,predictorNames,classNames)

t = templateSVM('Standardize',false,'SaveSupportVectors',true,'BoxConstraint',box_constraint,...
     'KernelFunction',kernel,'KernelScale',kernel_scale,'ShrinkagePeriod',shrinkage_period);

 Mdl = fitcecoc(X_train,y_train_str,'Learners',t,'ResponseName',responseName,...
     'PredictorNames',predictorNames,'ClassNames',classNames);

 %Methods for cross validation
 %CVMdl = crossval(Mdl,'KFold',5);
 %kfoldLoss(CVMdl)

 %Predict
 train_predictions =Mdl.predict(X_train);
 test_predictions = Mdl.predict(X_test);
 
 %convert test predictions back to intergers
 train_predictions_str =  strrep(train_predictions,"Unknown","-1");
 train_predictions_str =   strrep(train_predictions_str,"Non-Phishing Event"," 0");
 train_predictions_str =   str2num(cell2mat(strrep(train_predictions_str,"Phishing Event"," 1")));
 test_predictions_str =  strrep(test_predictions,"Unknown","-1");
 test_predictions_str =   strrep(test_predictions_str,"Non-Phishing Event"," 0");
 test_predictions_str =   str2num(cell2mat(strrep(test_predictions_str,"Phishing Event"," 1")));

 %Build confusion matrix
 train_confusion = confusionmat(train_predictions_str,y_train);
 test_confusion = confusionmat(test_predictions_str,y_test);
 overal_confusion = confusionmat([train_predictions_str;test_predictions_str],[y_train;y_test]);
 
 %Build Accuracy
 train_accuracy = sum(train_predictions == y_train_str) ./ size(y_train,1);
 test_accuracy = sum(test_predictions == y_test_str) ./ size(y_test,1);
end

