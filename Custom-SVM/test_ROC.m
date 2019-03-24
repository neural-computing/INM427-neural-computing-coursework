% [X1,Y1] = perfcurve(y_train,train_predictions{ind(1)},1)
% [X2,Y2] = perfcurve(y_train,train_predictions{ind(1)},0)
% [X3,Y3] = perfcurve(y_train,train_predictions{ind(1)},-1)
% 
figure(11)
% hold on
% plot(X1,Y1)
% plot(X2,Y2)
% plot(X3,Y3)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')
% hold off


% load simplecluster_dataset
% net = patternnet(20);
% net = train(net,simpleclusterInputs,simpleclusterTargets);
% simpleclusterOutputs = sim(net,simpleclusterInputs);
% plotroc(simpleclusterTargets,simpleclusterOutputs)

plotroc(y_train,train_predictions{ind(1)},"test")