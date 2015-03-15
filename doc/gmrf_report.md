# Gaussian Markov Random Fields

## Background

Gaussian Markov Random Field (GMRF) can be seen as a graph whose nodes follow a Gaussian distribution. Hence, a GMRF with $n$ nodes can be defined by a multivariate Gaussian:

$$\pi(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{n/2}}exp(\frac{-(x - \mu)^T\Sigma^{-1}(x - \mu)}{2})$$

The distribution has parameters $\mu$ a $n$-dimensional mean vector and the $n\times n$ covariance matrix $\Sigma$.

Moreover, GMRF exhibits probabilistic dependence between pair of nodes linked by an edge. Conversely, two nodes are conditionally independent given all the others if there is no edge between them.
The structure of the graph can be inferred from the precision matrix $Q=\Sigma^{-1}$. And nodes $i$ and $j$ are conditionally independent when $Q_{ij} = 0$. Therefore $Q$ is key in defining a GMRF whose density can be expressed as:

$$\pi(x) = \frac{1}{(2\pi)^{n/2}}|Q|^{n/2}exp(\frac{-(x - \mu)^TQ(x - \mu)}{2})$$

GMRF have been extensively used to model spatial, temporal as well as spatio-temporal data. For instance, GMRF has been used to predict rainfall data on a finer scale than the observations[@GMRFRainFall]. The structure of the graph is given by considering adjacent geographical area. The parameters of the spatio-temporal model were filled by minimising the difference between the expected and observed correlation.

## Graph Lasso
Since $Q$ describes the structure of the graph, it is inherently sparse. The graph lasso algorithm aims to estimate $Q$ by minimising a function.

$$-log(det Q) + tr(SQ) + \lambda||Q||_1$$

The regularisation parameter $\lambda$ encourages zero entries in $Q$. $||Q||_1$ is the $l1$-norm of $Q$ and $S$ is the empirical covariance matrix. The above function is convex but non-smooth since $l1$-norm is not differentiable[@Murphy]. The function is usually minimised by coordinate gradient descent. The regularisation parameter can be estimated by cross-validation.

## Analysis
We want to estimate the precision matrix $Q$ using Graph lasso. The model is then scored using the Bayesian Information Criterion (BIC):

$$BIC(Q, X) = ln(\sum N(X_i | \mu, Q)) - \frac{D}{2}ln(n)$$

where $X$ is the dataset, $Q$ the precision matrix, $\mu$ the empirical mean vector, $n$ the number of data points and $D$ the number of parameters.
Since the precision matrix is symmetrical, we have:

$$D = {n \choose 2} + n$$

We repeat the procedure for three different datasets: one with only the racks temperature, another with the racks as well as the outlet temperature of the AHUs and a third with the racks temperature, the outlets as well as the IT power consumption at the room level.

We plot the score of the graphs produced by the glasso algorithm given the regularisation parameters $\lambda$, see Figure \ref{scores}.

![BIC score against $\lambda$ \label{scores}](scores2.png)

From those plots, we can see that adding the outlets temperature as well as the IT power consumption adds significant information and increases the score dramatically. Given those information the score is maximised by $\lambda=0.2$

This graph has 43 nodes and 500 edges which shows that while the graph is not fully connected, it also lacks sparsity.
We can plot the connectivity matrix of the best graph (see Figure \ref{con}), where black cells represent edges. We can see that the graph exhibits spatial dependence (eg Q13-Q25 with N10-N25) but it also has longer edges which does not encode spatial relationship (eg EO6 connected to N2 and Q1).

![Connectivity matrix of the best graph \label{con}](con_matrix.png)

# References
