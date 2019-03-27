# Title
## INM427 Coursework

Mark Longhurst & Thomas Martin

29th March 2019

Repo: 

## Abstract

This paper reports on a critical evaluation of two machine learning models in the task of identifying phishing websites. The models considered are a feed-forward neural network SVM classifier. ... Summary of methodology used ... ... Summary of evaluation techniques ... ... Which model performed better? ... 

TODO: Check name of models

## I. Introduction

Phishing websites represent a subset of the wider phishing problem. In general, phishing relates to any attempt to fraudulently obtain sensitive personal information in an electronic communication. Phishing websites are websites that trick users into believing they are on an otherwise legitimate website, typically using a range of frontend web technologies in order to gain this information [1]. This is an increasingly important problem due to the increasing reliance on web-based services.

This paper aims to evaluate the performance of two models in the task of identifying phishing websites based on 9 features. The models considered in this paper belong to feed-forward neural networks and support vector machine (SVM) family of models. We use cross-validation to perform hyperparameter tuning from the parameter space.

The paper is organised as follows, section 2 provides an exploratory overview of the dataset used in the project, section 3 details the approach taken in the project to train and compare the models, section 4 discusses the results of the previous process, with section 5 providing a final conclusion.

### II.I Feed-Forward Neural Network (FNN)

Feed-forward neural networks refer to the generalised, multilayer perceptron model. These model consist of nodes or neurons arranged in input and output layers, typically with one or more hidden layers in between. Individual nodes are connected by edges called "weights", which denote the strength of the relationship between any node pair. During the training process, these weights are adjusted following an algorithm such as gradient descent, which is a systematic process of determine the impact a given node had on the output produced in a forward pass.

TODO: Maybe change discussion of model training?

They are high performing models in supervised tasks especially where the dataset is large, high dimensional, and unstructured.

### II.II Support Vector Machine (SVM)

Support vector machine (SVM) classifiers determine hyperplane of maxmimum margin separating the classes of the dataset. SVM is appropriate for finding nonlinear boundaries, even in case of high-dimensional datasets using a kernel function. Can be generalised to non-binary classification problems by following either one-vs-one or one-vs-all classification algorithm.

Compared to a feed-forward neural network, SVM classifiers operate similarly to a shallow neural network. SVM are generally thought to produce more easily understandable models.

## II. Dataset

The dataset used in this study was taken from the UCI Machine Learning Repository, originally collected from the Phishtank data archive [3]. The dataset contains features corresponding to 1353 websites, classed as either legitimate, suspicious, or phishy, encoded as 1, 0, and -1 respectively. There is a slight imbalance between these classes occurring with a frequency of 548, 702 and 103 for legitimate, phishy and suspicious samples respectively. However, this imbalance is not dramatic enough to require additional sampling methods.

??? Require considering phishy and suspicious together

The values for each feature are scaled to -1, 0, and 1 so no further feature engineering was required.

TODO: summary table of dataset

| Class               | Legitimate (548)                 | Phishy (702)                     | Suspicious (103)                 |
| ------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| Predictor Variables | Legitimate | Suspicious | Phishy | Legitimate | Suspicious | Phishy | Legitimate | Suspicious | Phishy |
| ------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| SFH                 |
| popUpWidnow         |
| SSLfinal_State      |
| Request_URL         |
| URL_of_Anchor       |
| web_traffic         |
| URL_Length          |
| age_of_domain       |
| having_IP_Address   |

### II.I Exploratory Data Analysis

## III. Methods

Initially a fully connected feed-forward neural network was built utilizing the sigmoid function as the neuron activation function. 

![Alt text](./diagrams/Feed-Forward-Diagram.png?raw=true "Feed Forward Neural Network")

The following hyperparameter were tuned:
* n_hidden_nodes - number of fully connected nodes within the hidden layer
* learning_rate - rate at which the models weights are updated during the back-propagation
* momentum - level of inertia added when modifying the models weights
* early_stopping_thresh - smallest delta allowed when updating the models weights before training is halted


Next a multiclass SVM model was constructed using MATLAB's "fitcecoc" method. 

The following hyperparameter were tuned:
* kernel
* box_constraint
* kernel_scale
* shrinkage_period


Cross validation was employed with 5 folds to try and give a better understanding of how well the data models would generalize despite the relatively small datasets.

COnfusion matrices were ploted to allow us to compare to accuracy of the classifiers
## IV. Results
128 models were run for both the Neural Net and SVM.

Below are the result:

### Top 10 configurations for SVM.

| box constraint | kernel scale | shrinkage period | kernel | train accuracy    | test accuracy     | cross fold error  | kfold accuracy    |
|----------------|--------------|------------------|--------|-------------------|-------------------|-------------------|-------------------|
| 0.65           | 1            | 10               | rbf    | 0.939809926082365 | 0.876847290640394 | 0.112342941611236 | 0.887657058388766 |
| 0.95           | 1            | 10               | rbf    | 0.953537486800422 | 0.881773399014778 | 0.115299334811531 | 0.88470066518847  |
| 0.65           | 1            | 1                | rbf    | 0.939809926082365 | 0.876847290640394 | 0.116038433111605 | 0.883961566888396 |
| 0.95           | 1            | 7                | rbf    | 0.953537486800422 | 0.881773399014778 | 0.116038433111605 | 0.883961566888396 |
| 0.95           | 1            | 4                | rbf    | 0.953537486800422 | 0.881773399014778 | 0.116777531411679 | 0.883222468588322 |
| 0.95           | 1            | 1                | rbf    | 0.953537486800422 | 0.881773399014778 | 0.117516629711753 | 0.882483370288248 |
| 0.65           | 1            | 7                | rbf    | 0.939809926082365 | 0.876847290640394 | 0.118994826311901 | 0.8810051736881   |
| 0.95           | 0.7          | 4                | rbf    | 0.960929250263992 | 0.876847290640394 | 0.123429416112344 | 0.876570583887657 |
| 0.65           | 1            | 4                | rbf    | 0.939809926082365 | 0.876847290640394 | 0.127864005912788 | 0.872135994087214 |
| 0.95           | 0.7          | 7                | rbf    | 0.960929250263992 | 0.876847290640394 | 0.129342202512936 | 0.870657797487066 |

### Top 10 Configurations for NN.

| n hidden node | learning rate | momentum | early stopping thresh | mean(train accuracy k) | mean(test accuracy k) | mean(time taken k) |
|---------------|---------------|----------|-----------------------|------------------------|-----------------------|--------------------|
| 26            | 0.046         | 0.005    | 0.001                 | 0.949629629629629      | 0.892592592592593     | 5.34200000000001   |
| 26            | 0.046         | 0.03     | 0.001                 | 0.939814814814815      | 0.892592592592593     | 4.18000000000002   |
| 34            | 0.046         | 0.005    | 0.001                 | 0.949814814814815      | 0.892592592592592     | 4.62199999999993   |
| 34            | 0.031         | 0.03     | 0.001                 | 0.945185185185185      | 0.891111111111111     | 5.52799999999997   |
| 18            | 0.046         | 0.03     | 0.001                 | 0.941111111111111      | 0.891111111111111     | 4.27600000000002   |
| 34            | 0.046         | 0.03     | 0.001                 | 0.943518518518518      | 0.891111111111111     | 3.96800000000003   |
| 30            | 0.046         | 0.005    | 0.001                 | 0.949074074074074      | 0.89037037037037      | 4.56399999999994   |
| 22            | 0.046         | 0.005    | 0.001                 | 0.942777777777778      | 0.888888888888889     | 3.77399999999998   |
| 30            | 0.046         | 0.03     | 0.001                 | 0.945925925925926      | 0.888888888888889     | 4.41800000000003   |
| 22            | 0.031         | 0.005    | 0.001                 | 0.943148148148148      | 0.888148148148148     | 5.47600000000002   |


### Confusion Matrices for the best model in each:
Neural Net
![Alt text](../Custom-NN/best_NN_kfold_test_confusion_compare_this.png?raw=true "5-fold confusion matrix for Best Neural Net Model")

SVM
![Alt text](../Custom-SVM/best_SVM_kfold_test_confusion_compare_this.png?raw=true "5-fold confusion matrix for Best SVM Model")

## V. Conclusion

## VI. References

[1] Phishing Web Site Methods https://www.webcitation.org/5w9Z2iACi?url=http://www.fraudwatchinternational.com/phishing-fraud/phishing-web-site-methods/
[3] https://archive.ics.uci.edu/ml/datasets/Website+Phishing
[4] 
