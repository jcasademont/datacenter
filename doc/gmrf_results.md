# Gaussian Markov Random Fields

## Data

4Energy provided us with a dataset of several variables spanning over March 2014. From that dataset, we select the variables which seem the most useful to predict the temperature of the different racks.

The selected variables are:

* The temperature of each racks identified by an alphanumeric ID such as E12, H9, N1...
* The temperature of the air blown by the four air handling units (AHU)
* The I.T. power consumption measured at the datacenter level, it is the sum of the consumption of each rack.

The racks and AHUs temperature are given at a 15 minutes interval, whereas the I.T. power consumption is measured every hour. We use linear interpolation on the power consumption data so that all variables have a value every 15 minutes.

We also assume that the AHUs temperature are controlled directly and therefore does not need to be predicted.

## Model

The graph lasso algorithm estimate the precision matrix[@Glasso] of the Gaussian Markov Random Field by minimising:

$$-log(det Q) + tr(SQ) + \alpha||Q||_1$$

Since our dataset is formed of time series, our model must display spatial but also temporal dependencies. Therefore we choose to include the following variables in our model:

* All the racks temperature at time $t$ and $t - 1$ denoted as $R_t$ and $R_{t-1}$
* The AHUs variables noted as $A_t$ and $A_{t-1}$
* The I.T. power consumption noted as $P_t$ and $P_{t-1}$

### Model selection

In order to select the hyperparameter $\alpha$ of the graph lasso, we use the Bayesian Information Criterion (BIC) where:

$$BIC(Q, X) = -2 * ln(\sum N(X_i | \mu, Q)) + D * ln(n)$$

where $X$ is the dataset, $Q$ the precision matrix, $\mu$ the empirical mean vector, $n$ the number of data points and $D$ the number of parameters.

We want the model with the highest log-likelihood and the lowest number of parameters. Therefore the best model correspond to the lowest BIC value.

### Data transformation

GMRF assumes that the data are jointly Gaussian, we can make sure that this condition is met by transforming the variables so that they are normally distributed. One such transformation from X to Z is:

$$Z = \Phi^{-1}(F(X))$$

where $\Phi$ is the cumulative distribution function of the normal distribution, $N(0, 1)$, and $F$ is the empirical cdf of X [@xue2012].

Since the transformed data is jointly Gaussian, we can infer the underlying graph from the precision matrix: $Q_{ij} = 0$ if and only if there is no edge between $Z_i$ and $Z_j$. Moreover we can prove that the graph of the original data has the same structure as the graph of the transformed variables [@xue2012].

### Results

We run the graph lasso on the original data and on the transformed data, for different values of $\alpha$. We plot the results in Figure \ref{bics}.

![BIC score against $\alpha$ for gaussian and non-gaussian data \label{bics}](bics.pdf)

Surprisingly, the Gaussian data score is slightly worse than the original data. We can also see that the selected hyperparameter is $\alpha=0.1$.

The best model gives the connectivity matrix displayed in Figure \ref{con_matrix}.

![Connectivity matrix of the model using the untransformed data and $\alpha=0.1$ \label{con_matrix}](con_matrix.pdf)

From the connectivity matrix, we can see that the graph lasso recovers a coherent structure. Indeed, most of the variables are dependent on their previous value. Moreover, we can see spatial structure becoming apparent such as EO6 - E9 - E12 or H15 - H19 - H22 - H25.

## Predictive accuracy

The BIC score gives us a relative mean of comparing different models. Now that we have selected our model, we need to check its predictive accuracy.

### One step accuracy

We first check its accuracy on predicting value of the next time step. That is, given the values at $t - 1$ as well as the values of the AHUs at $t$ we want the racks temperature and the power consumption at $t$, which can be written as $R_t, P_t | R_{t-1}, P_{t-1}, A_t, A_{t-1}$.

From [@GMRFbook], we know that, if $x \sim N(\mu, Q)$ and we partition $x$, $\mu$ and $Q$ as follow:

$$X = \begin{pmatrix} x_A \\ x_B \end{pmatrix},\ \mu = \begin{pmatrix} \mu_A \\ \mu_B \end{pmatrix} \text{ and } Q = \begin{pmatrix} Q_{AA} & Q_{AB} \\ Q_{BA} & Q_{BB} \end{pmatrix}$$

we have :
$$x_A | x_B \sim N(\mu_A - Q^{-1}_{AA}Q_{AB}(x_B - \mu_B), Q_{AA})$$

Since the most likely value of a Gaussian random variable is its mean, we use the estimation $x_A = \mu_A - Q^{-1}_{AA}Q_{AB}(x_B - \mu_B)$.
Thus we reorder the mean vector and the precision matrix so that $x_A = \{R_t, P_t\}$ and $x_B = \{R_{t-1}, P_{t-1}, A_t, A_{t-1}\}$.

Using this result, we perform a 5-fold cross validation and measure the absolute deviation between the estimate and the actual value. Figure \ref{box} shows the mean of those deviations across the five folds for all the predicted variables.

![Estimation errors for the racks temperature and the power consumption \label{box}](box.pdf)

The results show clearly that the prediction error for one time step, in other word 15 minutes, is reasonable even when taking into account the maximum deviation.

### $n$-steps accuracy

However, we may need to do predictions on several time steps. To do so, after estimating $R_t$ and $P_t$ we reuse that estimation to estimate the next time step $R_{t+1}$ and $P_{t+1}$. We always assume that we know the values of the AHUs at all time steps.

Figure \ref{n_steps} shows the mean absolute deviation between the estimate and the actual values across all the time steps in the 5-fold cross-validation.

![Mean Absolute Deviation against time steps \label{n_steps}](n_steps.pdf)

By zooming on the first 100 time steps (see figure \ref{n_steps_zoom}), we can see that the mean absolute deviation is around 1°C when predicting 25 hours in advance.

![Zoom focusing on the first 100 time steps \label{n_steps_zoom}](n_steps2.pdf)

\clearpage

# References
