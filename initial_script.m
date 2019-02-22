% Read in Raw csv
raw = csvread("PhishingData.csv",1);

%split raw features from targets
features = raw(:,1:9);
target = raw(:,10);

%one hot encode target variables for use within the matlab neural network
%toolset, we add 2 to the target variables as well so that our target
%classes are 1 2 and 3 rather than -1 0 and 1.
target = ind2vec(target'+2)';
