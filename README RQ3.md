# Question 3

**How does the decision tree regressor model compare to the k-nearest neighbour regressor model in terms of in- and out-of-sample error for time series forecasting?**

Decision tree regression (DTR) and K-neirest neighbor (KNN) are two techniques for solving regression problems. They, however, differ vastly in how their regression hypothesis is obtained. DTR makes its predictions based on a model it learned, while KNN does not learn a model at all, the data is the model. 

Let us try to understand the workings of both algorithms in more detail.



# Decision tree regression

Decision trees have implementations both for classification (classification trees) and for regression (regression trees). In what follows we will explain the workings of the regression case. The decision tree algorithm produces a model that can be used for predictions. Given an input $\bold{x}$, the model obtains a prediction $\hat y$ by following a series of decision rules. 

The model can be represented as a tree, where each **internal node** corresponds to a decision. Internal nodes have one incoming edges and out-going edges, labeled with the choise of a decision. The **leaf nodes** represent the final outcome of the perdiction ($\hat y$). In the classification case, those outcomes are categorical, while in the regression case, the leaves represent a numeric value.

By starting from the **root node** and following the path to a leaf node, a $\hat y$ value can be obtained given an input $\bold{x}$. The image below displays an example of a regression tree.

<img src="https://www.researchgate.net/profile/Fritz-Schiltz-2/publication/325712556/figure/fig1/AS:636637654552580@1528797653733/Example-of-a-regression-tree-Panel-a-with-the-partition-of-the-covariates-space-Panel.png" alt="Example of a regression tree (Panel a), with the partition of the... |  Download Scientific Diagram" style="zoom:50%;" />

Source: [(PDF) Using Regression Tree Ensembles to Model Interaction Effects: A Graphical Approach (researchgate.net)](https://www.researchgate.net/publication/325712556_Using_Regression_Tree_Ensembles_to_Model_Interaction_Effects_A_Graphical_Approach/figures?lo=1)



Let us now invastigate how the algorithm works to obtain its regression trees. We will explain this by means of an example. We must predict the weight of people given their height, shoe size and gender. 
$$
\bold{x} = \text{<?height, ?shoe size, ?gender>}
\newline
y = \text{?weight}
$$
To build the decision tree, the algorithm must first decide on what its root node should be. Lets assume a node contains the following structure: "X is less than y", where X corresponds to a feature of the dataset and y corresponds to a certain treshold. An example node could then be: " *weight is less than 45*".  When deciding upon a node, there are two questions to ask, 1)  What should the threshold be? and 2) What feature should we use? (eg: weight or shoe size) 

1. **What should the threshold be?**

   A plot of the height in relationship to the weight may look something like this:

   ![image-20220602194437933](C:\GitHub\SFML\image-20220602194437933.png)

   Source: [The Relationship Between Human Height and Weight – GeoGebra](https://www.geogebra.org/m/RRprACv4)

   

   To select the optimal threshold, the data is split into two halfs, from each half the mean is calculated. For each data point the distance is calculated to its respective mean, this is called the residual. By squaring those residuals and summing them up the squared residuals are obtained. The objective is to minimize the sum of squared residuals. We demonstrate this in the following example.

   In the graph, a potential treshold could for example be right in the middle at height = 55 inches. Splitting the data into two parts at height = 55 and calculating the mean weight values of the data points on the left side and the mean of the data poitns on the right side may look as follows:

   ![image-20220602195415563](C:\GitHub\SFML\image-20220602195415563.png)

   The sum of squared residuals on the left side will be rather small for the left side, however the sum of squared residuals on the right side will be larger. This is because sum of the y-distance to the left mean is smaller than the sum of y-distances to the right mean. If we calculate the sum of the left and right residuals, we may not have minimized it in this case.

   However, if we had selected a different threshold (eg height = 65), the sum of squared residuals on the left side may be a little bit higher, but the total sum of squared errors (left + right) will be lower! This is thus a better threshold.

   ![image-20220602195934856](C:\GitHub\SFML\image-20220602195934856.png)

    The algorithm selects the threshold that minimizes the total sum of squared errors.

   

2. **What feature should we use?**

   Similar to what we have done to obtain the perfect threshold, the best feature is selected by trying out different thresholds and selecting the feature that has the smallest sum of squared residuals. 

   Note that this strategy also works for categorical values. To obtain the best sum of squared residuals for gender for example, there is only one threshold to be tested.

   ![image-20220602203133222](C:\GitHub\SFML\image-20220602203133222.png)



**Overfitting**

The default behaviour of the decision tree regressor is to keep splitting the data into smaller groups until they can not be split any further. (Eg: all data points that correspond to a particular leave have the same value as the mean). This will cause the model to perfectly fit the training data, but the model will be severly overfit. This overfitting can be mitigated by setting a minimum on the number of data points that belong to a particular group. What is often done for example, is to set the minimum number to 20 for each leave. If a node has less than 20 data points, it will not be split further and instead take the mean of those data points as its output. By doing we, we limit how deep branches of the tree can grow.

The image below displays the predictions of two regression tree models that were trained on a noisy sine wave. What we can see that the result of decision tree regression is not a smooth continuous function such as a polynomial, but instead a function with many straight edges. This comes from the fact that similar data points are grouped the average of their target values is taken. Here we can also see how limiting the depth of the tree, will cause more elements to be grouped together, which prevents overfitting.



![../_images/sphx_glr_plot_tree_regression_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)

Source: [1.10. Decision Trees — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/tree.html#tree)



Although, model predictions are really fast (logarithmic in the number of data points used to train the tree), one of the disadvanges of decision trees are that they take a long time to train. Learning an optimal decision tree is an NP-complete problem and thus decision-tree learning is based on greedy heuristics algorithms that do not guarantee a global optimal solution. We will see that K-NN mitigates this problem by not training a model at all.

Source: [1.10. Decision Trees — scikit-learn 1.1.1 documentation](https://scikit-learn.org/stable/modules/tree.html#tree)



# KNN regression

Just like decision trees, KNN can be used for both classification and regression. KNN, however, differs vastly from decision trees in the way the predictions are obtained. Instead of training a model and letting the model do the predications, in KNN the data is the model. 

KNN has the advantage that it is a lot simpler than decision trees. The idea behind KNN is that given an input $\bold{x}$, the target value $y$ is obtained by looking at all k data points in $\bold{x}$'s neighborhoud. This neighborhood contains all the k data points that are the most similar to $\bold{x}$. The output of the KNN algorithm depends on whether it is doing classificaiton or regression.  In the case of classification, the target value for $\bold{x}$ is obtained by selecting the mode of the target values in its neighborheid. In the regression case, however, the average value of the neighborhood's target values is selected as output. 

The image below visually illustrates the workings of the KNN algorithm. In the example k = 1 and thus the neighborhood only contains one point. The algorithm would thus select the target value of the star within the circle as its output for $\bold{x}$.

![image-20220602211346211](C:\GitHub\SFML\image-20220602211346211.png)

Source: [KNN Classification Tutorial using Sklearn Python | DataCamp](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn)





The distance can be calculated via different metrics:

- **Minkowski Distance**
- **Euclidean Distance**
- **Manhattan Distance**



if the features are categorical,





Closest points:

- categorical features, vs real valued features



```
KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. The size of the neighbourhood needs to be set by the analyst or can be chosen using cross-validation (we will see this later) to select the size that minimises the mean-squared error.

While the method is quite appealing, it quickly becomes impractical when the dimension increases, i.e., when there are many independent variables.

https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html
```



Argument why probably not so good results: I can imagine appliance values to either be high or low, so maybe the mean of 2 mountains may not be such a good strategy.

values has many extremes









bs: We are given a data set with the temperatures of different rooms (eg: kitchen, bathroom, bedroom, etc) in a particular house at different timestamps. **Our task is to predict the humidity in the living room** at a given timepoint. This is the same regression problem we will tackle to compare the two algorithms later.

