# How does KNN-regression compare to Decision tree regression in terms of training/testing error for time series data



K-neirest neighbor (KNN) and Decision tree regression (DTR) are two techniques for solving regression problems. They, however, differ vastly in how their regression hypothesis is obtained.  

ne learns model, other does not



# Decision tree regression



# KNN regression

Just like decision trees, KNN can be used for both classification and regression. KNN, however, differs vastly from decision trees in the way the predictions are obtained. Instead of training a model and letting the model do the predications, KNN the data becomes the model.

The way classification is done is  

most commonly used for classification. 

KNN is much simpler in how it obtains its regression line.

Closest points:

- categorical features, vs real valued features



```
KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. The size of the neighbourhood needs to be set by the analyst or can be chosen using cross-validation (we will see this later) to select the size that minimises the mean-squared error.

While the method is quite appealing, it quickly becomes impractical when the dimension increases, i.e., when there are many independent variables.

https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html
```



Argument why probably not so good results: I can imagine appliance values to either be high or low, so maybe the mean of 2 mountains may not be such a good strategy.

values has many extremes