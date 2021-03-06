function [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,train_predictions_str,...
    test_predictions_str,cross_fold_error,validationPredictions,kfold_confusion,kfold_accuracy] = SVM_wrapper(X,y,y_explode,X_train,y_train,X_test,...
    y_test,y_train_str,y_test_str,kernel,box_constraint,kernel_scale,shrinkage_period,responseName,predictorNames,classNames)
                
t = templateSVM('Standardize',false,'SaveSupportVectors',true,'BoxConstraint',box_constraint,...
     'KernelFunction',kernel,'KernelScale',kernel_scale,'ShrinkagePeriod',shrinkage_period);

 Mdl = fitcecoc(X_train,y_train_str,'Learners',t,'ResponseName',responseName,...
     'PredictorNames',predictorNames,'ClassNames',classNames);
 
 Mdlcv = fitcecoc(X,y_explode,'Learners',t,'ResponseName',responseName,...
     'PredictorNames',predictorNames,'ClassNames',classNames);
 
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

 %Methods for cross validation
 CVMdl = crossval(Mdlcv,'KFold',5);
 cross_fold_error = kfoldLoss(CVMdl);
 [validationPredictions, validationScores] = kfoldPredict(CVMdl);
 validationPredictions_str =  strrep(validationPredictions,"Unknown","-1");
 validationPredictions_str =   strrep(validationPredictions_str,"Non-Phishing Event"," 0");
 validationPredictions_str =   str2num(cell2mat(strrep(validationPredictions_str,"Phishing Event"," 1")));
 kfold_confusion = confusionmat(validationPredictions_str,y);
 
 %Build Accuracy
 train_accuracy = sum(train_predictions == y_train_str) ./ size(y_train,1);
 test_accuracy = sum(test_predictions == y_test_str) ./ size(y_test,1);
 kfold_accuracy = sum(validationPredictions_str == y) ./ size(y,1);
end

