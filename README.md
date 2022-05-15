# Support vector machine (SVM)

Similar to the Perceptron Learning Algorithm (PLA), SVM is meant to classify binary data by separating data in two groups. However, instead of trying to a hyperplane that works (like PLA), SVM calculates the hyperplane that best separates the data. Its objective is thus to find the hyperplane that maximizes margin between the two groups. (It is thus an optimization problem)

![image-20220514162339201](C:\GitHub\SFML\image-20220514162339201.png)

The way this largest margin is achieved, is by maximizing the distance between the hyperplane and the closest data point to the line. This distance can be expressed as the following equation:
$$
\frac{1}{\mid\mid \bold{w}\mid\mid}
$$
which can be transformed to a minimization problem by minimizing the following equation:
$$
\frac{1}{2}\bold{w}^T\bold{w}
$$
Subject to the following constraints
$$
yn(wTxn + b)
$$
**[Lagrange formulation]**

Formulating the minimization problem as a Lagrange formulation. Via quadratic programming the vector $\bold{\alpha}$ can be calculated, which allows us to obtain 
$$
\bold{w} = \sum_{\bold{x_n} \in SV}{\bold{\alpha_n}y_n \bold{x_n}}
$$


**Non linear data**

Although this SVM approach works well for linearly separable data, non linearly separable data must be transformed to an other dimension to make it separable. An example of this is show on the image below.

![image-20220514170712418](C:\GitHub\SFML\image-20220514170712418.png)

The first drawback of this approach is that someone must chose what the non linear transformation should be. The second problem is that when a complex decision boundary is required, the complexity of the transformation increases, which increases computational requirements. The kernel trick is meant to solve these issues. When we look at the Lagrange equation below, via quadratic programming the alpha's can be obtained. A dot product must be calculated for each data point in the transformed space. The higher the dimension of the space, the more complex the computation becomes. The kernel trick prevents having to calculate this dot product.
$$
\mathcal{L}(\bold{\alpha}) = \sum_{n=1}^{N} \alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N} y_ny_m\alpha_n\alpha_m\bold{z_n}^T \bold{z}_m
$$




algor for svm doesn't need to know what each point is mapped to and its non-linear transformation. all how points are compared to each other after applying non linear transformation. This corresponds to taking the dot product between the two points in their transformed space. It is known as the kernel function:
$$
k(x, x') = z^Tz'
$$
Polynomial kernel:


$$

$$






Via a 

Issues:

must calculate inproduct from Lagrange thing, kernel trick allows to mitigate this and go to infinite dimensionality.









In what follows next, we investigate the following research question: "*How do do the linear kernel, polynomial kernel and Gaussian radial basis function (RBF) compare to each other, when applied to a synthetic two-dimensional dataset*?"









```
from sklearn import svm
X = [
[ 0.1, 2,3 ], ...
]

y = [0, 1, 0, ...]

svm.SVC().fit(X, y)
```





Sources:

[The Kernel Trick in Support Vector Machine (SVM) - YouTube](https://www.youtube.com/watch?v=Q7vT0--5VII)