<div>
  <script type="text/x-mathjax-config">
    MathJax = {
      tex: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$'], ['\\[','\\]']]
      }
    };
  </script>
  <script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>
</div>

## 3.7 Ordinary Least Squares

The OLS method is a foundational technique for linear regression. This method aims to provide the best possible estimates of the regression coefficients, facilitating the prediction of the behavior of a dependent variable based on independent variables. It works by minimizing the sum of the squares of the differences between observed and predicted values, aiming to make these predictions as accurate as possible.

Mathematically, the goal of OLS can be expressed through the sum of squared residuals (SSR), which is the sum of squared differences between the observed values (\\( y_t \\)) and the predicted values (\\( \hat{y}_t \\)). The sum of squared residuals is therefore expressed as:

$$ SSR = \sum_{t=1}^T (y_t - \hat{y}_t)^2 $$

where the model is given as:

$$ \hat{y}_t = \beta_0 + \beta_1 X_{t,1} + \beta_2 X_{t,2} + \cdots + \beta_n X_{t,n} + \epsilon $$

Where, \\( \hat y_t \\) is the dependent variable (e.g., energy consumption), \\( X_{t,1}, X_{t,2}, \ldots, X_{t,n} \\) are independent variables (such as weather conditions), \\( \beta_0 \\) is the intercept, \\( \beta_1, \beta_2, \ldots, \beta_n \\) are the coefficients that need to be estimated, and \\( \epsilon \\) is the error term (Hastie et al., 2017).

The essence of OLS is to find the best estimates for these coefficients, \\( \hat{\beta} \\), that minimize the SSR. \\( \hat{\beta} \\) is found by solving the OLS normal equations, and the solution can be shown using matrix notation as:

$$ \hat{\beta} = (X^T X)^{-1} X^T Y $$

In this equation, \\( X \\) is the matrix of independent variables, \\( X^T \\) is the transposed matrix of independent variables, and \\( Y \\) is the vector of the observed values.

Another essential application for understanding OLS is the Gauss-Markov theorem in linear models, as it establishes the conditions under which OLS estimators are considered the Best Linear Unbiased Estimators (BLUE). This distinction means that among all linear estimators that are unbiased (their expected value is equal to the true parameter value), OLS estimators have the smallest variance, making them the most precise or efficient.

### The Gauss-Markov Assumptions

For OLS estimators to be BLUE, the linear regression model must satisfy the following assumptions:

1. **Linearity:** The model is linear in parameters, which can be represented as:

$$ Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_k X_{ik} + \epsilon_i $$

where \\( Y_i \\) is the dependent variable, \\( X_{1i}, \ldots, X_{ik} \\) are the independent variables, \\( \beta_0 \\) is the intercept, and \\( \beta_1, \ldots, \beta_k \\) are the parameters (coefficients), and \\( \epsilon_i \\) is the error term for the \\( i \\)-th observation.

2. **Independence:** The residuals (errors) \\( \epsilon_i \\) are independent of each other.

3. **Homoscedasticity:** The variance of the residuals is constant across all observations:

$$ \text{Var}(\epsilon_i) = \sigma^2 $$

for all \\( i \\) where \\( \sigma^2 \\) is constant.

4. **No Autocorrelation:** The covariance between any two residuals is zero:

$$ \text{Cov}(\epsilon_i, \epsilon_j) = 0 $$

for all \\( i \neq j \\).

### The BLUE Property

Under these conditions, the OLS estimator \\( \hat{\beta} \\) for the coefficients \\( \beta \\) minimizes the sum of squared residuals, providing the best linear unbiased estimate. The formula for the OLS estimator in matrix form is given by:

$$ \hat{\beta} = (X^T X)^{-1} X^T Y $$

This formula represents the estimation of the coefficients that minimizes the difference between observed and predicted values, thereby ensuring efficiency and unbiasedness as guaranteed by the Gauss-Markov theorem.

### Implications of the Gauss-Markov Theorem

The assurance that OLS estimators are BLUE relies on the Gauss-Markov assumptions. If these conditions are violated—for instance, if the residuals exhibit heteroscedasticity or autocorrelation—the efficiency and unbiasedness of the OLS estimators can be compromised. Such violations make adjustments to the model or the application of alternative estimation techniques necessary to regain the desirable properties of the estimators (Hastie et al., 2017).

### 3.7.1 Modelling Deseasonalized and Detrended Components with OLS

In regression analysis, understanding and modeling residuals is crucial for evaluating model performance and identifying potential improvements. Residuals, the differences between observed and predicted values, can be analyzed using the OLS method to gain further insights into data patterns and model shortcomings. This approach involves treating the residuals as a dependent variable in a new OLS model, to predict the residuals from a set of predictors.

Formulating the Model for Residuals

Consider the residuals \\( R_t \\) from an initial regression model. We propose modeling these residuals using OLS with a set of predictors \\( X_t^T = (X_{t,1}, X_{t,2}, \ldots, X_{t,p}) \\), leading to the residual model:

$$ \hat{R_t} = \beta_0 + \beta_1 X_{t,1} + \beta_2 X_{t,2} + \cdots + \beta_p X_{t,p} + \epsilon_t $$
$$ = \beta_0 + \sum_{i=1}^p X_{t,i} \beta_i + \epsilon_t $$

where \\( \beta_0 \\) represents the intercept, \\( \beta_i \\) are the coefficients for each predictor, \\( X_{t,i} \\) denotes the independent variables at time \\( t \\), and \\( \epsilon_t \\) is the error term for the residual model.

The aim is to minimize the sum of squared residuals of this model:

$$ \min_{\beta_i} \sum_{t=1}^T (R_t - \beta_0 - \sum_{i=1}^p X_{t,i} \beta_i)^2 $$

By doing so, the model attempts to explain the pattern of residuals left unaddressed by the initial regression, potentially uncovering systematic variations missed earlier (Hastie et al., 2017, pp. 44-49).
