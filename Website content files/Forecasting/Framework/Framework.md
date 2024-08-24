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

## 5.1 Framework

This section of the thesis will outline the framework adopted to forecast energy consumption across the customer groups. The primary dataset comprises time series data where \\( Y_t \\) represents the consumption variable (dependent variable) and \\( X_t \\) denotes a set of independent variables influencing consumption. A critical preprocessing step involves the application of MSTL, for decomposing the time series data to eliminate underlying trend and seasonality from \\( Y_t \\) , returning the remainders \\( R_t \\) , these remainders represent the detrended and deseasonalized consumption data. Chapter 3.1.2.6 offers a more detailed description of MSTL.

Categorical variables were transformed into dummy variables and normalizing numerical variables through min-max scaling. Independent variable \\( X_t \\) are then lagged to \\( X_t-h \\) , where \\( h \\) is the horizon, to facilitate prediction into the future.

The data is then split into training and test sets, where a sliding rolling window approach is employed. The model is retrained in a specified interval, \\( m \\) , where the lower \\( m \\) is the higher the retraining frequency is. It uses a static window size, \\( n_train \\) , for the training data. The established model (OLS and RF for this thesis) then fits \\( R_train~X_train \\) , and the model is then used to forecast the remainders 𝑚 steps ahead. After each forecasting step, the window rolls forward, using the static training window size, to be retrained and to predict once again.

To estimate the consumption, the trend and seasonal component is added back onto the estimated remainder, \\( \hat{R_t} \\) , which would then provide the estimated energy consumption for each timestep, \\( \hat{Y_t} \\) .
The framework used is meant to predict the detrended and deseasonalized energy consumption efficiently, while handling the volatile nature of the energy consumption timeseries and the individual characteristics of each customer group.


```python

```
