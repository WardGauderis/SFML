# Support vector machine (SVM)

Similar to the Perceptron Learning Algorithm (PLA), SVM is meant to classify binary data by separating data in two groups. However, instead of trying to create a hyperplane that works, SVM calculates the hyperplane that best separates the data. Its objective is thus to find the hyperplane that maximizes margin between the two categories. (It is thus an optimization problem)

![image-20220514162339201](C:\GitHub\SFML\image-20220514162339201.png)

The way this largest margin is achieved, is by maximizing the distance between the hyperplane and the closest data point to the line. This distance can be expressed as the following equation:
$$
\frac{1}{\mid\mid \bold{w}\mid\mid}
$$
which can be expressed as a minimization problem via the following equation:
$$
\frac{1}{2}\bold{w}^T\bold{w}
$$
Subject to the following constraints (assuming the data is linearly separable):
$$
y_n(\bold{w}^T\bold{x_n} + b) >= 1
$$


The minimization problem is solved by formulating it as a Lagrange minimization problem and by applying quadratic programming to obtain the vector $\bold{\alpha}$. This vector $\bold{\alpha}$ enables us to compute the weights as follows:
$$
\bold{w}= \sum_{x_n \in \text{ SV}}{\alpha_n y_n \bold{x_n}}
$$


**Non linear data**

Although the default SVM approach works well for linearly separable data, non linearly separable data must be transformed to an other dimension to make it separable. An example of such data is shown in the image below.

![image-20220514170712418](C:\GitHub\SFML\image-20220514170712418.png)

The first drawback of these space transformations is that someone must chose what the non linear transformation should be. The second problem is that when a complex decision boundary is required, the complexity of the transformation increases, which increases computational requirements. The kernel trick is meant to solve these issues. When we look at the Lagrange equation below, via quadratic programming the alpha's can be obtained. A dot product must be calculated for each data point in the transformed space. The higher the dimension of the space, the more complex the computation becomes. The kernel trick prevents having to calculate this dot product, allowing for transformations to infinite dimensional space.
$$
\mathcal{L}(\bold{\alpha}) = \sum_{n=1}^{N} \alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N} y_ny_m\alpha_n\alpha_m\bold{z_n}^T \bold{z}_m
$$



In what follows next, we investigate the following research question: "*How do do the linear kernel, polynomial kernel and Gaussian radial basis function (RBF) compare to each other, when applied to a synthetic two-dimensional dataset*?"




Linear kernel, which is equivalent to the default SVM explained above:
$$
K(\bold{x}, \bold{x}') = \bold{z}^T \bold{z}'
$$
Polynomial kernel:


$$
K(\bold{x}, \bold{x}')=(1 + \bold{x}^T\bold{x}')^Q
$$

Radial basis function:
$$
K(\bold{x}, \bold{x}') = e^{-\gamma \mid\mid \bold{x}-\bold{x}' \mid\mid^2}
$$


----------------------------



To demonstrate the difference between the three kernels, we have opted for a non linear dataset that is separable. We use the following 2 dimensional spiral dataset.



-----

Sklearn implements SVM using a soft margin, however, since the synthetic dataset is actually separable a strict margin is preferred. To obtain this stricter margin, we must provide a high C value. When C goes to infinity, we again, obtain the strict margin. Changing the C value corresponds to changing the C value in the following minimization problem
$$
\frac{1}{2}\bold{w}^T\bold{w} + \sum_{n=1}^{N} \zeta_n
$$
Where 
$$
\sum_{n=1}^{N} \zeta_n
$$


corresponds to the total violation. This is a quantification for the points that inside the margin, and thus violate the margin.

<img src="https://miro.medium.com/max/916/1*UylttRyv51Pz0ADkNt4goA.png" alt="Support Vector Machine Explained. Theory, Implementation, and… | by Zixuan  Zhang | Towards Data Science" style="zoom:33%;" />



Indeed, when C goes to infinity, this is equal to solving the minimization problem for the strict margin. (Which was discussed above)
$$
\frac{1}{2}\bold{w}^T\bold{w}
$$


----



There is a clear difference in the results between the default SVM (linear kernel) approach and the two kernels (RBF, Polynomial). 

All that the linear kernel is able to do separate the data using one linear line. It is clearly not able to fit the data.

In the case of the polynomial kernel, by default Sklearn uses a number of degrees equal to three. This value corresponds to the $Q$ in the polynomial kernel mentioned above. The higher the number of degrees, the more flexible the decision boundary becomes. 

Although a degree of three, clearly does not manage to separate the data, its decision boundary is still different to the linear decision boundary. We try higher degrees of freedom later. Note that the higher the degrees, the more prone the model is to overfitting.

From the third graph above, we can see that the RBF kernel manages to fit the decision boundary really well.

**sv's**?

---



Lastly, we have tried some different values for the polynomial kernel. We have tried the following degrees: 5, 10, 15, 20, 30 and none of them managed to fit the data decently. 



---

It is also interesting to look at the number of support vectors that are required to obtain the decision boundary. This has great implications for generalization. Using the formula
$$
\mathbb{E}[E_{out}] \le \frac{\mathbb{E}[\text{\# of SVs}]}{\text{N} - 1}.
$$

---

In the case of the SVM, assuming that  $\mathbb{E}[\text{\# of SVs}] = 17$, we can say the expected value of $E_{out}$ is less than or equal to 4.2%. This is a great result! The linear and polynomial kernels do not perform this well on the generalization bound. This is due to their high number of SVs.

Note: Please keep in mind that the assumption that $\mathbb{E}[\text{\# of SVs}] = 17$ is not entirely correct. To obtain more precise results, the number of SVs should be calculated multiple times.







# Sources:

[The Kernel Trick in Support Vector Machine (SVM) - YouTube](https://www.youtube.com/watch?v=Q7vT0--5VII)