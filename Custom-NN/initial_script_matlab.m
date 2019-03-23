format short
more off

clear all;
% Read in Raw csv
raw = csvread("../data/PhishingData.csv",1); %needed for matlab
%raw = csvread("PhishingData.csv");

%Set parameters
n_hidden_nodes=10:5:60;
n_classes = 3; 
weights_scale_factor = 0.1;
learning_rates=0.001:0.005:0.1;
epochs = 150;
momentums=0.005:0.025:0.05;
early_stopping_threshs=0.005:0.025:0.05;
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
wbm = waitbar(0,sprintf("Training Models (%f)", total_tests));
confusions = cell(total_tests,n_classes,n_classes);

%how stable is the network? we see that 
for learning_rate=learning_rates
    for momentum=momentums
        for early_stopping_thresh=early_stopping_threshs
            
            %[confusion,train_accuracy,test_accuracy,time_taken] = pararrayfun(nproc, fun, n_hidden_nodes,"UniformOutput", false);
            for n_hidden_node=n_hidden_nodes
                tests = tests+1;
                waitbar((tests/total_tests),wbm,"Training Models");
                [confusion,train_accuracy,test_accuracy,time_taken,accuracy_per_epoch,error_per_epoch] = NN_model(X_train,y_train,X_test,y_test,y_train_explode,n_hidden_node,n_classes,weights_scale_factor,learning_rate,epochs,momentum,early_stopping_thresh,early_stopping_resilance,debug);
                display([tests,test_accuracy])
                overal_results(tests,:) = [n_hidden_node,learning_rate,momentum,early_stopping_thresh,train_accuracy,test_accuracy,time_taken];
                confusions{tests} = confusion;
            end
        end
    end
end

close(wbm);

%pick best test
[v,ind] = maxk(overal_results(:,6),5);
overal_results(ind,:)
confusions{ind}

%test stability
best_nn = overal_results(ind(1),:);

%test stability of top model - when the seed for matlabs random number
%generator is fixed, we get the same output. Otherwise we see variation.
best_distro = zeros(20,3);
for i=1:20
    [confusion,train_accuracy,test_accuracy,time_taken,accuracy_per_epoch,error_per_epoch] = NN_model(X_train,y_train,X_test,y_test,y_train_explode,...
        best_nn(1),n_classes,weights_scale_factor,best_nn(2),...
        epochs,best_nn(3),best_nn(4),early_stopping_resilance,debug);
    best_distro(i,:) = [i,train_accuracy,test_accuracy];
    disp([i,test_accuracy])
    
end
disp(best_distro);

%these are randomly sampled events so we can model as a normal distribution
[m,st] = normfit(best_distro(:,2:end));

%and plot to see variation within the distributions
figure(1)
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

%plot accuracy and error per epoch
figure(2)
hold on
x = 1:size(accuracy_per_epoch,1);
plot(x,accuracy_per_epoch)
plot(x,error_per_epoch)
legend('Training Accuracy per Epoch','Training Error per Epoch')
hold off

writematrix(overal_results,"NN_model_results.csv")
writematrix(best_distro,"NN_best_distro.csv")

saveas(figure(1),"Test-NN.png")
saveas(figure(2),"Test-NN-acc-err.png")


