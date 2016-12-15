# kaggle-bcwd

Given a set of 30 features, the classification task here is to determine whether the cancer is benign or malignant. 

### Data source
The origina datasource is the one found on the UCI repository [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)) which has 10 features for each cell nucleus, namely,
a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The data has been modified by Kaggle by adding in more features such as the mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius. The details about this could be found [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

For more details, please check the project and the data description [here](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names).

The class distribution is as follows:
- 357 benign
- 212 malignant

### Approach:

We implement an active learner as well as a random learner using AdaBoost classifier with a Decision Tree as an estimator. For the active learner, we select points of uncertainity by choosing distances whose difference from the decision boundary is minimum (in this case, we use probability estimates given by the classifier). The random learner chooses points at random for the classification task. 

We start with an initial batch size of randomly selected points for both the classifiers, and vary the batch sizes and the number of total available calls to the oracle. We did not perform a grid search of the parameters for the classifer, since that would defeat the purpose of comparing the active learner with the random learner. 

We experimented with SVM as the base learner with linear kernel, and observed that the AdaBoost with DT as estimator performed slightly better. SVM with RBF as kernel performed worst by classifying all labels as benign, but that's probably because we used the default C and gamma values. We leave the parameter tuning part by grid search over the parameter space for the future update.

### Results:
The following results were observed with fixed random seeds so that we can replicate the results. Even if the random seed is not initialized, the results were pretty much similar.

#### Predictions for the active learner :
```
Classification report for classifier AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=400, random_state=None):
             precision    recall  f1-score   support

          B       1.00      0.98      0.99       124
          M       0.97      1.00      0.98        64

avg / total       0.99      0.99      0.99       188
```
The confusion matrix is given as follows:
```
Confusion matrix:
[[122   2]
 [  0  64]]
 ```
#### Predictions for the random learner :
 ```
Classification report for classifier AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=400, random_state=None):
             precision    recall  f1-score   support

          B       0.99      0.97      0.98       124
          M       0.94      0.98      0.96        64

avg / total       0.97      0.97      0.97       188
```
The confusion matrix is given as follows:
```
Confusion matrix:
[[120   4]
 [  1  63]]
```

As we see in the plot below, the active learner clearly outperforms the random learner. We see the best results when the active learner just used 270 queried labels with an accuracy of 98.94%  (error rate = 0.0106).
![alt text](https://github.com/MascarenhasV/kaggle-bcwd/blob/master/plots/plot.png "Active vs. Random Learner")

In almost all cases (without random seeds), the active learner did better than the random learner. We also keep track of the best model in case the active learner overfits after achieving the best results.


