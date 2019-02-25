pkg load nnet
page_output_immediately (1);

t=cputime()

clear all;
% Read in Raw csv
%raw = csvread("PhishingData.csv",1);
raw = csvread("PhishingData.csv");

n_hidden_nodes = 25;
n_classes = 3;
weights_scale_factor = 0.2;
learning_rate = 0.005;
epochs = 5000;
momentum = 0.0001;

%split data into test and train sectionsq
p = .7;      % proportion of rows to select for training
N = size(raw,1); % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;     
tf = tf(randperm(N));   % randomise order
dataTraining = raw(tf,:); 
dataTesting = raw(~tf,:);

%split raw features from targets
X_train = dataTraining(:,1:9);
y_train = dataTraining(:,10);
%clear raw;

%one hot encode target variables for use within the matlab neural network
%toolset, we add 2 to the target variables as well so that our target
%classes are 1 2 and 3 rather than -1 0 and 1.
y_train = full(ind2vec(y_train'+2)');


function g = sigmoid(z)
  g = sigmoid = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGradient(z)
  g = sigmoid(z) .* (1 - sigmoid(z));
end

function w = buildWeights(i_size, j_size, epsilon = 1)
	w = rand(j_size, i_size) * 2 * epsilon - epsilon;
  % creates a matrix of j_size rows and i_size columns within +- epsilon
end

%Initialize bias as additional column vector on the front 
%of the feature matrix, ready for matrix multiplication of weights
d1=0;
d2=0;
disp("training...")
for iter=1:epochs
  waitbar(iter/epochs)
  for m=1:size(X_train,1)
    
    a1 = [1;X_train(m,:)']; # 10x1  a1 is a column vector for a sample +1 (for bias)
    
    %Next we initialize some weights, calculate the input potential matrix for the hidden layer and 
    % then calculate the hidden layer activations and gradients
    if iter == 1 && m==1
      theta1 = buildWeights(size(a1,1),n_hidden_nodes,weights_scale_factor);
      disp("weights1 initialized");
    end
    z2 = [1;theta1 * a1]; % 26x1 calculate input potential of hidden layer
    a2 = sigmoid(z2); % 26x1 calculate activation of hiddent layer
    g2 = sigmoidGradient(z2); % 26x1 calculate gradients with these activations
    
    if iter == 1 && m==1
      theta2 = buildWeights(size(a2,1),n_classes,weights_scale_factor);
      disp("weights2 initialized");
    end
    z3 = theta2 * a2; % compute input potential of output layer
    a3 = sigmoid(z3); % compute activation of output layer 
    g3 = sigmoidGradient(z3); % compute gradients at output layer - possibly not needed

    %back prop - calculate errors through network
    e3 = a3-y_train(m,:)'; % calculate errors at output layer
    e2 = theta2' * e3 .* g2; % calculate errors at hidden layer 

    %calculate deltas for each layer
    if iter == 1 && m==1
      d2 = -learning_rate * (e3*a2'); % calculate deltas for theta2
      d1 = -learning_rate * (e2*a1'); % calculate deltas for theta1 
    else
      d2 = -learning_rate * (e3*a2') + momentum * d2; % calculate deltas for theta2
      d1 = -learning_rate * (e2*a1') + momentum * d1; % calculate deltas for theta1 
    end
    
    theta2 = theta2 + d2;
    theta1 = theta1 + d1(2:end,:);
    %disp(g2);
    %disp(sum(sum(d1)));
    %disp(sum(sum(d2)))
  end
end

disp("finished training");

predictions = [];
%split raw features from targets on test data
X_test = dataTesting(:,1:9);
y_test = dataTesting(:,10);

for m=1:size(X_test,1);
  a1 = [1;X_test(m,:)']; # a1 is a column vector for a sample +1 (for bias)

  %Next we initialize some weights, calculate the input potential matrix for the hidden layer and 
  % then calculate the hidden layer activations and gradients
  z2 = [1;theta1 * a1]; % 26x1 calculate input potential of hidden layer
  a2 = sigmoid(z2); % 26x1 calculate activation of hiddent layer
  z3 = theta2 * a2;
  a3 = sigmoid(z3);
  [val, idx] = max(a3);
  predictions = [predictions;idx-2];
end
 
results = [predictions y_test];

function [C,rate]=confmat(Y,T)
  %CONFMAT Compute a confusion matrix.
  %
  %	Description
  %	[C, RATE] = CONFMAT(Y, T) computes the confusion matrix C and
  %	classification performance RATE for the predictions mat{y} compared
  %	with the targets T.  The data is assumed to be in a 1-of-N encoding,
  %	unless there is just one column, when it is assumed to be a 2 class
  %	problem with a 0-1 encoding.  Each row of Y and T corresponds to a
  %	single example.
  %
  %	In the confusion matrix, the rows represent the true classes and the
  %	columns the predicted classes.  The vector RATE has two entries: the
  %	percentage of correct classifications and the total number of correct
  %	classifications.
  %
  %	See also
  %	CONFFIG, DEMTRAIN
  %

  %	Copyright (c) Ian T Nabney (1996-2001)

  [n c]=size(Y);
  [n2 c2]=size(T);

  if n~=n2 | c~=c2
    error('Outputs and targets are different sizes')
  end

  if c > 1
    % Find the winning class assuming 1-of-N encoding
    [maximum Yclass] = max(Y', [], 1);

    TL=[1:c]*T';
  else
    % Assume two classes with 0-1 encoding
    c = 2;
    class2 = find(T > 0.5);
    TL = ones(n, 1);
    TL(class2) = 2;
    class2 = find(Y > 0.5);
    Yclass = ones(n, 1);
    Yclass(class2) = 2;
  end

  % Compute 
  correct = (Yclass==TL);
  total=sum(sum(correct));
  rate=[total*100/n total];

  C=zeros(c,c);
  for i=1:c
    for j=1:c
      C(i,j) = sum((Yclass==j).*(TL==i));
    end
  end   
end

%show results in a confusion matrix
%results matrix has to be one hot encoded to work.
confusion = confmat(full(ind2vec(results(:,1)+2)'),full(ind2vec(results(:,2)+2)'));
disp(cputime - t)