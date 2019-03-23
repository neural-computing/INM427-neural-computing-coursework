# Title
## INM427 Coursework

Mark Longhurst & Thomas Martin

29th March 2019

Repo: 

## Abstract

This paper reports on a critical evaluation of two machine learning models in the task of identifying phishing websites. The models considered are ... some neural network ... and random forest. ... Summary of methodology used ... ... Summary of evaluation techniques ... ... Which model performed better? ... 

TODO: Check name of models

## I. Introduction

Phishing websites represent a subset of the wider phishing problem. In general, phishing relates to any attempt to fraudulently obtain sensitive personal information in an electronic communication. Phishing websites are websites that trick users into believing they are on an otherwise legitimate website, typically using a range of frontend web technologies in order to gain this information [1]. This is an increasingly important problem due to the increasing reliance on web-based services.

This paper aims to evaluate the performance of two models in the task of identifying phishing websites based on 9 features. The models considered in this paper belong to feed-forward neural networks and random forest family of models. ... something about cross validation ... 

The paper is organised as follows, section 2 provides an exploratory overview of the dataset used in the project, section 3 details the approach taken in the project to train and compare the models, section 4 discusses the results of the previous process, with section 5 providing a final conclusion.

### II.I Feed-Forward Neural Network (FNN)

Feed-forward neural networks refer to the generalised, multilayer perceptron model. These model consist of nodes or neurons arranged in input and output layers, typically with one or more hidden layers in between. Individual nodes are connected by edges called "weights", which denote the strength of the relationship between any node pair. During the training process, these weights are adjusted following an algorithm such as gradient descent, which is a systematic process of determine the impact a given node had on the output produced in a forward pass.

TODO: Maybe change discussion of model training?

They are high performing models in supervised tasks especially where the dataset is large, high dimensional, and unstructured.

### II.II Random Forest (RF)

Random forest models and an ensemble technique, which can be applied to a classification task. It can be understood as aggregating the result of many "shallow" decision trees, such that the final error will be much smaller than a single equivalent decision tree.

The main advantages of random forest models are they typically able to avoid overfitting as well as handle missing values and outliers without prior data preprocessing. Compared to neural networks, random forests can often boast better performance for more structured, lower dimensional datasets.

## II. Dataset

The dataset used in this study was taken from the UCI Machine Learning Repository, originally collected from the Phishtank data archive [3]. The dataset contains features corresponding to 1353 websites, classed as either legitimate, suspicious, or phishy, encoded as 1, 0, and -1 respectively. There is a slight imbalance between these classes occurring with a frequency of 548, 702 and 103 for legitamate, phishy and suspicious samples respectively. However, this imbalance is not dramatic enough to require additional sampling methods.

??? Require considering phishy and suspicious together

The values for each feature are scaled to -1, 0, and 1 so no further feature engineering was required.

TODO: summary table of dataset

### II.I Exploratory Data Analysis

## III. Methods

## IV. Results

## V. Conclusion

## VI. References

[1] Phishing Web Site Methods https://www.webcitation.org/5w9Z2iACi?url=http://www.fraudwatchinternational.com/phishing-fraud/phishing-web-site-methods/
[3] https://archive.ics.uci.edu/ml/datasets/Website+Phishing