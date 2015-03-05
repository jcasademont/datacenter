# Gaussian Markov Random Fields

## Background

Gaussian Markov Random Field (GMRF) can be seen as a graph whose all nodes follow a Gaussian distribution. Hence, a GMRF with $n$ nodes can be defined by a multivariate Gaussian:

$$\pi(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{n/2}}exp(\frac{-(x - \mu)^T\Sigma^{-1}(x - \mu)}{2})$$

The distribution has parameters $\mu$ a $n$-dimensional mean vector and the $n\times n$ covariance matrix $\Sigma$.

Moreover, GMRF exhibits probabilistic dependence between pair of nodes linked by an edge. Conversely, two nodes are conditionally independent given all the others if there is no edge between them.
The structure of the graph can be inferred from the precision matrix $Q=\Sigma^{-1}$. And nodes $i$ and $j$ are conditionally independent when $Q_{ij} = 0$. Therefore $Q$ is key in defining a GMRF and its density can be expressed as:

$$\pi(x) = \frac{1}{(2\pi)^{n/2}}|Q|^{n/2}exp(\frac{-(x - \mu)^TQ(x - \mu)}{2})$$

GMRF have been extensively used to model spatial, temporal as well as spatio-temporal data. (ADD LITERATURE HERE).

## Graph Lasso
Since $Q$ describes the structure of the graph, it is inherently sparse. The graph lasso algorithm aims to estimate $Q$ by minimising a function.

$$-log(det Q) + tr(SQ) + \lambda||Q||_1$$

The regularisation parameter $\lambda$ encourages zero entries in $Q$. $||Q||_1$ is the 1-norm of $Q$ and $S$ is the empirical covariance matrix.
(Convex but not smooth -> norm not differential, cross-validation to determine $\lambda$ + insight from Stanford paper)

## Analysis
