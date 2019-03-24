format short
more off

clear all;
% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab

%Set parameters
n_hidden_nodes=10:4:40; %60;
n_classes = 3; 
weights_scale_factor = 0.2;
learning_rates=0.001:0.015:0.0501%0.101;
epochs = 150;
momentums=0.005:0.025:0.05;
early_stopping_threshs=0.001:0.05:0.1;
early_stopping_resilance = 10;
test_train_split_p = .7;      % proportion of rows to select for training
debug = false;
tests = 0;

%build test/train data
[X_train,y_train,X_test,y_test,y_train_explode] = process_data(raw,test_train_split_p);
  
%initialize some more parameters  
total_tests = size(learning_rates,2) * size(momentums,2) * size(early_stopping_threshs,2) * size(n_hidden_nodes,2);
disp(total_tests);
overal_results = zeros(total_tests,7);
wbm = waitbar(0,sprintf("Training Models (%d)", total_tests));
train_confusions = cell(total_tests);
test_confusions = cell(total_tests);
overal_confusions = cell(total_tests);
train_predictions = cell(total_tests);
test_predictions = cell(total_tests);

%how stable is the network? we see that 
for learning_rate=learning_rates
    for momentum=momentums
        for early_stopping_thresh=early_stopping_threshs
            
            for n_hidden_node=n_hidden_nodes
                
                tests = tests+1;
                waitbar((tests/total_tests),wbm,sprintf("Training Models (%d/%d)", [tests,total_tests]));
                
                [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,time_taken,accuracy_per_epoch,error_per_epoch,train_results,test_results]...
                    = NN_model(X_train,y_train,X_test,y_test,y_train_explode,n_hidden_node,n_classes,weights_scale_factor,learning_rate,epochs,...
                    momentum,early_stopping_thresh,early_stopping_resilance,debug);
                
                display([tests,test_accuracy])
                
                overal_results(tests,:) = [n_hidden_node,learning_rate,momentum,early_stopping_thresh,train_accuracy,test_accuracy,time_taken];
                
                train_confusions{tests} = train_confusion;
                test_confusions{tests} = test_confusion;
                overal_confusions{tests} = overal_confusion;
                train_predictions{tests} = train_predictions;
                test_predictions{tests} = test_predictions;

            end
        end
    end
end

close(wbm);

%pick best test
[v,ind] = maxk(overal_results(:,6),5);
overal_results(ind,:)
test_confusions{ind}

%test stability
best_nn = overal_results(ind(1),:);

%test stability of top model - when the seed for matlabs random number
%generator is fixed, we get the same output. Otherwise we see variation.
best_distro = zeros(20,3);
for i=1:20
    [train_confusion,test_confusion,overal_confusion,train_accuracy,test_accuracy,time_taken,accuracy_per_epoch,error_per_epoch,train_results,test_results]...
        = NN_model(X_train,y_train,X_test,y_test,y_train_explode,best_nn(1),n_classes,weights_scale_factor,best_nn(2),...
        epochs,best_nn(3),best_nn(4),early_stopping_resilance,debug);
    
    best_distro(i,:) = [i,train_accuracy,test_accuracy];
    
    disp([i,test_accuracy])
    
end
disp(best_distro);

%these are randomly sampled events so we can model as a normal distribution
[m,st] = normfit(best_distro(:,2:end));

%and plot to see variation within the distributions
figure(1)
grid on
grid minor
hold on
x = [0.8:0.001:1];
y1 = normpdf(x,m(1),st(1));
y2 = normpdf(x,m(2),st(2));
plot(x,y1)
plot(x,y2)
legend('Train Distribution','Test Distribution');
xlabel("Accuracy");
ylabel("Probability Density for 'Best' Model Accuracy");
hold off

%plot accuracy and error per epoch
figure(2)
hold on
x = 1:size(accuracy_per_epoch,1);
plot(x,accuracy_per_epoch)
legend('Training Accuracy per Epoch','Location','East');
xlabel("Epochs");
ylabel("Training Accuracy");
grid on
grid minor
hold off

% figure(3)
% hold on
% x = 1:size(accuracy_per_epoch,1);
% plot(x,error_per_epoch)
% legend('Training Error per Epoch','Location','East');
% xlabel("Epochs");
% ylabel("Overal Sum of Absolute Error whilst training");
% grid on
% hold off

disp("saving diagrams")
saveas(figure(1),"Test-NN-Prob-Dens.png")
saveas(figure(2),"Test-NN-acc.png")
%saveas(figure(3),"Test-NN-err.png")

disp("saving datasets")
writematrix(overal_results,"NN_model_results.csv")
writematrix(best_distro,"NN_best_distro.csv")
writematrix(overal_results(ind,:),"NN_top_10.csv")

disp("saving  confusion plots")
%plot and save confusion matrix for best model
labels = {'Phishing', 'Non-Phishing', 'Unknown'};
plot_confusion(train_confusions{ind(1)},labels,"Training Set",4)
plot_confusion(test_confusions{ind(1)},labels,"Testing Set",5)
plot_confusion(overal_confusions{ind(1)},labels,"Overal Set",6)
saveas(figure(4),"best_NN_train_confusion.png")
saveas(figure(5),"best_NN_test_confusion.png")
saveas(figure(6),"best_NN_overal_confusion.png")
disp("complete")
