# Tree-Based Learners

## Usage
Command: python driver.py Data/Istanbul.csv

## Introduction 
In this report, we will explore multiple tree-based supervised learning algorithms such as decision tree, random tree, bagged learner and insane learner (combining linear regression with bagging), and dive deeper into aspects of overfitting. We will explore how hyperparameter tuning as well as adjusting bagging will impact the performance of the model on an in-sample validation set versus an out-of-sample validation set, and in some cases leading to overfitting. Further on, we will conduct experiments to quantitatively measure metrics such as variance and training time in addition to RMSE (root mean squared error) associated with the overall performance of the learner. 

Before we dive deeper into the discussions posed above, let us discuss the learners implemented for these experiments in more detail: 

<ul>
<li><b>Decision Tree Learner:</b> For the purposes of this experiment, we have used JR Quinlan’s implementation of the decision tree algorithm outlined in his 1985 paper on Induction of decision trees. Quinlan’s paper talks about classification trees, however, we implement this decision tree to solve a regression problem, and hence use a median of the best feature to split on. The best feature is defined as the feature that has the highest absolute value correlation with the target variable. </li>
<li><b>Random Tree Learner:</b> Random tree learner is an alteration of JR Quinlan’s that chooses the feature to split on as a random feature instead of the feature with the highest correlation with the target variable. In addition, for selecting the split value it takes the average of two random data points for the randomly selected feature. This algorithm has been outlined by Adele Cutler. The idea for adding this randomness is to benefit in terms of speed of learning, and overcoming the randomness by increasing bagging and boosting by implementing forests of such random tree learners. </li>
<li><b>Bag Learner:</b> Bag learner is a bootstrap aggregating method for training multiple learners on different subsets / permutations of the provided data. The predictions from the ensemble of learners is aggregated to produce the final prediction of the bag learner. In the case of a classification problem, the prediction with the highest vote is returned. In the case of our experimentation for regression learners, the mean of the predictions is returned as the final prediction. </li>
<li>Insane Learner:</li> Insane learner is an ensemble of multiple bag learners with multiple linear regression learners (in our experiments we have used 20 bag learners and 20 linear regression learners for each bag learner). The idea for implementing the insane learner is to understand the power of combining ensemble techniques with computational power of modern computers to improve performance with multiple learners for a single prediction (regression) exercise. 

## Overfitting and Hyperparameter Tuning 
The discussion of overfitting and hyperparameter tuning will be carried out with the following specific question in mind: 
<i>Does overfitting occur with respect to leaf_size? Use the dataset Istanbul.csv with DTLearner. For which values of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting. Support your assertion with graphs/charts. (Don't use bagging). </i>

Overfitting is defined as the area of divergence where the in-sample error decreases and out-of-sample error increases. This happens when the learning algorithm is very closely fitted on the limited set of training data, and in consequence does not generalize well to the test (unseen) data points. 

For the purposes of our experimentation, we will be using the data provided in the Istanbul.csv1 file and the error metric to measure overfitting will be RMSE (root mean squared error) between the prediction and target variable.  The learner used for this experiment is a decision tree learner. 

### Experiment 1: Tuning hyperparameter in small increments to gauge Overfitting 
For this experiment, the leaf_size hyperparameter of the decision tree learner is tuned from 1 to 100 (since there ~540 records in the dataset), and the in-sample (train) rmse and the out-sample (test) rmse are recorded for each version of the decision tree. Leaf size is hyperparameter that is used to aggregate data points at the leaf level once a minimum threshold (specified by leaf size) has been reached. Hence intuitively, it is expected that the lower the leaf size, the more likely the tree is to overfit on the training dataset and less likely to perform just as well on the testing dataset. Hence the hypothesis is that leaf size is inversely proportional to overfitting. 


Above we see the chart of in-sample rmse versus out-of-sample rmse of decision tree learners as the leaf size of the learners is tuned from 1 to 100. As stated initially, the hypothesis that leaf size is inversely proportional to overfitting holds true. In other words, the lower the leaf size, the higher the overfitting (low in-sample rmse and high out-of-sample rmse). The point at which in-sample rmse and out-of-sample rmse is the same is between leaf size = 8 and leaf size = 9. On further zooming into the point of convergence, we notice the point of intersection is 8.75. The area to the left of the point of intersection is the area of overfitting. 

Hence, for leaf sizes less than 9, the decision tree learner tends to overfit on the Instanbul.csv data. Understanding the inverse relationship between leaf size and overfitting can sometimes seem counterintuitive. Here is alternate representation of the experiment 1 data where the x axis is inverted to show leaf size in decreasing order. 

For the above chart, it can be observed as the in-sample rmse decreases (goes to 0) and out-of-sample rmse increases as the leaf size decreases below 9.  

Hence from the above experimentation results, we can conclude that overfitting occurs as a result of decreasing leaf size of decision tree learners. Overfitting occurs specifically for values of leaf size less than 9 for the Instanbul.csv data. These results have been reported using RMSE, and the assertions have been supported by the graphs shown above. 

### Effects of Bagging on Overfitting 

The discussion of bagging and overfitting will be carried out with following question in mind: 

<i>Can bagging reduce or eliminate overfitting with respect to leaf_size? Again use the dataset Istanbul.csv with DTLearner. To investigate this choose a fixed number of bags to use and vary leaf_size to evaluate. Provide charts to validate your conclusions. Use RMSE as your metric. </i>

Bagging is the method of bootstrapping an ensemble of learners on different subsets / bags of the dataset and aggregating the output to generate a final prediction for the bagged learners. Bagging was designed as an ensemble meta-algorithm to reduce variance, improve stability and avoid overfitting. Intuitively, since bagged learners are trained on different bags of data (sampled with replacement), it does not tend to overfit on the nature of the entire training data. This intuition can be validated in the form of the following experimentation, where a bag learner comprising of 20 bags is used with decision trees of altering leaf sizes. 

For this experiment, we train a bag learner with 20 decision tree learners on 20 bags (samples with replacement) of the Istanbul.csv data. This parameter of bags is kept consistent across the experiment. Now, as per the previous experiment, the leaf size hyperparameter is tuned from 1 to 100 (since there are ~540 records in the data). Overfitting is gauged by measuring the in-sample rmse versus the out-of-sample rmse for each iteration of the bag learner. Since leaf size is the hyperparameter that aggregates data points at the leaf level, once a minimum threshold is reached (specified by leaf_size), it would seem intuitive as the leaf size decreases, overfitting would increase. However, bagging would help avoid this overfitting, since different samples of data would be used for different trees, and it would help overall learner generalize well over the test data. 

From the above graph, we observe that the bag learner with 20 decision trees also tends to overfit the data (low in-sample rmse and high out-of-sample rmse), however the threshold for overfitting seems to have been pushed further. In other words, bagging tries to avoid overfitting. On further zooming into the point of divergence of in-sample rmse and out-of-sample, we notice that the leaf size threshold for overfitting is still between 8 and 9. On taking a deeper look, it is visible that it reduced to 8.6 (as compared to 8.75 from the previous experiment).  

The above chart shows the zoomed in view of the intersection of in-sample rmse and out-of-sample rmse and the value of leaf size comes out to be 8.6. The area to the left of the point of intersection is the area of overfitting. In order to look at a more intuitive view of leaf size versus overfitting, the leaf size is plotted in decreasing order as shown in the graph below. 

The above graph shows that bagging causes the point of intersection between the in-sample rmse and out-of-sample rmse further to the right, thus reducing the area that represents overfitting in the above chart. 

Hence, based on the observations from experiment 2 we can conclude that bagging helps reduce overfitting with respect to leaf size. This observation was validated using the Instanbul.csv data, setting the number of bags to 20 and tuning the leaf size from 1 to 100 for decision tree learners. The associated graphs are provided in the experimentation above. 

### Comparison of Decision Tree Learner and Random Tree Learner 

The discussion of decision tree learner and random tree learner will be carried out with the following question in mind: 

Quantitatively compare "classic" decision trees (DTLearner) versus random trees (RTLearner). In which ways is one method better than the other? Provide at least two quantitative measures. Important, using two similar measures that illustrate the same broader metric does not count as two. (For example, do not use two measures for accuracy.) Note for this part of the report you must conduct new experiments, don't use the results of the experiments above for this(RMSE is not allowed as a new experiment). 

As mentioned in the introduction, decision tree learner has been implemented using JR Quinlan’s algorithm, while random tree learner has been implemented using Adele Cutler’s algorithm. There are two key differences between these two algorithms: 

Feature for split: Decision tree learners pick the best feature using a metric such as correlation, entropy or gini impurity that compares the feature against the target variable. For random tree learners, the feature is selected randomly from the available set of features. 

Split Value: For decision tree learners, the split value is calculated by taking the median of the best feature. For random tree learners, the split value is calculated by taking the average of two random data points along the randomly chosen feature. 

From this initial discussion, it seems like decision tree is better than a random tree in most aspects, however, random trees when applied over a large number of bags could provide significant performance benefits. We will discuss these benefits in further detail by measuring two quantitative measures – variance and training time for decision tree learners and random tree learners. 

### Experiment 3A – Variance 

Variance is defined as the expectation of the squared deviation of a random variable from its mean. In simpler terms, it measures how far a set of numbers are spread out from their average value. We conduct experiments to observe change in variance for decision tree learners as compared to random tree learners over different values of hyperparameters (leaf size). We plot the variance associated with the predicted values for decision tree as well as random tree learners for in-sample and out-of-sample datasets over leaf sizes varying from 1 to 100. 

The average in-sample variance for random tree (6.92E-05) is lower than the average in-sample variance for decision tree (8.13E-05). This generalizes to out-of-sample variance as well. The average out-of-sample variance for random tree (3.51E-05) is lower than the average out-of-sample variance for decision tree (3.84E-05). 

In order to understand if this variance also holds true for bag learners, we conducted an experiment for bag learners with 20 decision trees and 20 random trees for differing leaf sizes from 1 to 100. 

The observations for single tree learners generalize to bagged tree learners as well. The average in-sample variance for random tree (5.10E-05) is lower than the average in-sample variance for decision tree (6.42E-05). The average out-of-sample variance for random tree (2.48E-05) is lower than the average out-of-sample variance for decision tree (3.00E-05). It is interesting to observe how the variance reduces by a factor across decision trees as well as random trees as we go from single tree learners to bagged tree learners. 

Hence, using the variance metric we are able to notice that random trees tend to reduce the spread of the distribution of the predicted value from its mean. 

### Experiment 3B – Time 

The time to train the model is measured for decision tree learners as well as random tree learners. Intuitively, it would appear that random tree learners would be significantly faster since they randomize the most time intensive tasks of best feature selection and split value calculation. For single decision tree learners and random tree learners, the time for learning (model training) is measured in seconds over differing values of leaf sizes from 1 to 100. 

The average training time for a decision tree learner is 15 milliseconds, while the average training time for a random tree learner is 2.5 milliseconds. Thus, on an average a single random tree learner is 6 times faster than a decision tree learner. On conducting the same experiment for bagged learners with 20 decision trees and 20 random trees with differing leaf sizes from 1 to 100, the following results are observed: 

Results similar to single tree learners are observed for bagged tree learners as well. The average training time for a decision tree learner is 275 milliseconds, while the average training time for a random tree learner is 45 milliseconds. Thus, on an average a bagged random tree learner is 6 times faster than a bagged decision tree learner. 

Hence, from a learning time (model training) perspective, random tree learners operate much faster than decision tree learners. 

## Conclusion 

Over the course of this report we explored multiple tree-based learning algorithms and discussed the concepts of overfitting, leaf size tuning, bagging and quantitative metrics such as variance and time. In experiment 1, we noticed that as leaf size decreases, the decision tree learner tends to overfit the training data. In experiment 2, we observed that bagging helps reduce the impact of overfitting. In experiment 3, we noted that random tree learners perform better than decision trees from a variance and learning (model training) time perspective, as opposed to accuracy. Overall, this report helped explore multiple interesting topics for tree-based learners and opened doors to furthermore topics such as boosting, and combining different learners within the same bagged learner that we will explore at a later time. 