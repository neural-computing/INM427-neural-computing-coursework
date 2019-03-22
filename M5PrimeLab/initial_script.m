pkg load nnet
pkg load parallel
page_output_immediately (1);
format short
more off

test_train_split_p = 0.7
raw = csvread("../data/PhishingData.csv");

%split data into test and train sectionsq
N = size(raw,1); % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(test_train_split_p*N)) = true;     
tf = tf(randperm(N));   % randomise order
dataTraining = raw(tf,:); 
dataTesting = raw(~tf,:);

%split raw features from targets
X_train = dataTraining(:,1:9);
y_train = dataTraining(:,10);

%split raw features from targets on test data
X_test = dataTesting(:,1:9);
y_test = dataTesting(:,10);

%one hot encode target variables for use within the matlab neural network
%toolset, we add 2 to the target variables as well so that our target
%classes are 1 2 and 3 rather than -1 0 and 1.
y_train_explode = full(ind2vec(y_train'+2)');

# We willcreate a configuration for ensembles of regression trees: the minimum number of
% observations anode must have to be considered for splitting will be 5, the minimum number
% of trainingobservations a leaf node may represent will be 1, the trees will not be pruned,
% no smoothing will beapplied, and we will set splitThreshold equal to 1E-6.
params = m5pparams(false, 1, 5, false, 0, 1E-6);

paramsEnsemble = m5pparamsensemble(50, [], [], [], [], true, 0, false);

#identify binary features
isBinCat = [true(1,9)];

#number of features to use when generating trees
numVarsTry = [2 4 7 9];
figure;
hold on;
for i = 1:4
  paramsEnsemble.numVarsTry = numVarsTry(i);
  [~, ~, ensembleResults] = m5pbuild(X_train, y_train, params, isBinCat, paramsEnsemble);
  plot(ensembleResults.OOBError(:,1));
end
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');
legend({'2' '4' '7' '9'}, 'Location', 'NorthEast');

paramsEnsemble = m5pparamsensemble(200);
[model, time, ensembleResults] = m5pbuild(X_train, y_train, params, isBinCat, paramsEnsemble);

figure;
plot(ensembleResults.OOBError(:,1));
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');

figure;
bar(ensembleResults.varImportance(3,:) ./ ensembleResults.varImportance(4,:));
xlabel('Variable number');
ylabel('Variable importance');

figure;
contrib = ensembleResults.OOBContrib;
cminmax = [min(min(contrib(:,1:(end-1))))-0.5 max(max(contrib(:,1:(end-1))))+0.5];
for i = 1 : size(X_train,2)
  subplot(3,5,i);
  scatter(X_train(:,i), contrib(:,i), 50, '.');
  ylim(cminmax);
  xlim([min(X_train(:,i)) max(X_train(:,i))]);
  xlabel(['x_{' num2str(i) '}']);
  box on;
end

[Yq, contrib] = m5ppredict(model, X_test(1,:));
fprintf('Prediction: %f\n', Yq(1));
fprintf('In-bag mean: %f\n', contrib(1,end));
fprintf('Input variable contributions:\n');
[~, idx] = sort(abs(contrib(1,1:end-1)), 'descend');
for i = idx
  fprintf('x%d: %f\n', i, contrib(1,i));
end

rng(1);
resultsCV = m5pcv(X_train, y_train, params, isBinCat, 10, [], [], paramsEnsemble);
figure;
plot(ensembleResults.OOBError(:,1));
hold on;
plot(resultsCV.MSE);
grid on;
xlabel('Number of trees');
ylabel('MSE');
legend({'Out-of-bag' 'Cross-Validation'}, 'Location', 'NorthEast');