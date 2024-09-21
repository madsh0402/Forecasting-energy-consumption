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

## 5.6 Evalutation of the best prediction models

### 5.6.1 Modelling and comparing with the benchmark

Now that we have analyzed and found the optimal setting for each of the OLS and RF models, the effectiveness of the models in forecasting detrended and deseasonalized energy consumption can be evaluated, where the goal is to account for all variability not explained by trend or the seasonal components.

OLS, which by design has zero bias, directly minimizes the sum of squared errors between predicted and observed values, making it sensitive to outliers. This sensitivity means that extreme values in the data can influence the regression coefficient, which could prove to be a problem with MSTL remainders due to the possibility of seasonality not being perfectly captured at all timesteps. However, its simplicity and interpretability remain as strengths, especially when dealing with linear relationships where the number of predictors is not excessively large and its sparsity on computational resources allows for higher retraining frequency, which has been proven to have significant impact on accuracy.

Conversely, RF offers robustness to outliers and the ability to model complex, non-linear relationships between larger set of variables without requiring transformation or assumption of linearity. Unlike OLS, RF introduces some bias to gain a reduction in variance, benefitting from an ensemble of decision trees to improve accuracy and generalizability. This trade-off is particularly beneficial in handling the volatile energy consumption time series. However, to get an efficient trade-off between bias and variance, it is crucial to tune the parameters to the time series, which is extremely computationally demanding in both time and computational power. For this research, the parameters have been tuned for one customer group and reused for all customer groups which is suboptimal.

Both models’ performance will be evaluated on their predictive accuracy and compared to a benchmark. This baseline assumes that the MSTL decomposition perfectly accounts for trend and seasonality, leaving no remainder, meaning that a naive model predicts zero for the detrended and deseasonalized time series. This comparison helps in highlighting the additional variance each model can explain, thereby demonstrating their ability to capture the nuances of energy consumption beyond what can be attributed to predictable patterns alone.

#### 5.6.2 Comparing OLS with the benchmark model

Earlier in this chapter, the different settings of the framework used have been tested to find the most efficient settings for OLS, as seen earlier the OSL model performed best with a 1-year training window size, and the highest possible retraining frequency, which is every timestep.

To evaluate the accuracy of the forecasting models, Mean squared Error (MSE) will be used. It measures the average squared difference between the estimated values and the actual values as seen below:

$$ MSE = \frac{1}{T} \sum_{t=n_{train}+1}^{T} (R_t-\hat{R}_t)^2 $$

<details>
  <summary>Click to expand R code for the Diabold Marinao test</summary>

  <pre style="background-color: #f7f7f7; border: 1px solid #ddd; padding: 10px; overflow-x: auto; border-radius: 5px; font-size: 14px;">
  <code class="language-R">
# Forecsting remainders with OLS

## Libraries
library(tidyverse)
library(forecast)
library(ggplot2)
library(dplyr)
library(data.table)
library(IRdisplay)
library(progress)

library(foreach)
library(doParallel)

library(caret)
library(randomForest)
</code></pre>


<pre style="background-color: #f7f7f7; border: 1px solid #ddd; padding: 10px; overflow-x: auto; border-radius: 5px; font-size: 14px;">
<code class="language-R">
##################Setting workign directory and loadign data ###################
##### Setting workign directory and loadign data #####
base_path <- "Forecasting-energy-consumption"
base_path <- "Forecasting-energy-consumption"
setwd(base_path)
#data <- read.csv(paste0(base_path,"Data/Combined/Full_data_ecwap.csv"))
MSTL      <- fread(paste0(base_path,"/Data Cleaning/MSTL_decomp_results.csv"))
R_t_OLS   <- fread(paste0(base_path,"/Data/Results/OLS/R_hat_t/2yTrain_h=1_steps_ahead=1_OLS_R_hat_t.csv"))
R_t_RF    <- fread(paste0(base_path,"/Data/Results/RF/R_hat_t/h=1_steps_ahead=1_ntree=250_RF_R_hat_t.csv"))
</code></pre>


<pre style="background-color: #f7f7f7; border: 1px solid #ddd; padding: 10px; overflow-x: auto; border-radius: 5px; font-size: 14px;">
<code class="language-R">
  R_t_vec_1year <- tail(MSTL$Remainder, n=8759)
  R_t_vec_2year <- tail(MSTL$Remainder, n=17519)
  R_t_0_vec     <- MSTL$Null_Remainder
  R_t_OLS_vec   <- R_t_OLS$x
  R_t_RF_vec    <- R_t_RF$x

  e1 <- R_t_vec_2year-tail(R_t_0_vec, n=17519)
  e2 <- R_t_vec_1year-R_t_OLS_vec
  e3 <- R_t_vec_2year-R_t_RF_vec

  MSE_0      <- mean(e1^2)
  MSE_OLS    <- mean(e2^2)
  MSE_RF     <- mean(e3^2)
</code></pre>
</details><br>


```R
# Create a data frame (table)
mse_table <- data.frame(
  Model = c("MSE_0", "MSE_OLS"),
  MSE_Value = c(MSE_0, MSE_OLS)
)

# Display the table
print(mse_table)
```

        Model MSE_Value
    1   MSE_0  31514.66
    2 MSE_OLS  40865.14


From the table above, the OSL model has a MSE of 40865.14, higher than the median MSE of 31514.66 for the benchmark. This difference in MSE suggests that the naïve model performs better than the OLS model, meaning that it would be better to not even predict the remainders than using an OLS model.

To ensure that the lower MSE observed for the benchmark is not a result of randomness. The DM test offers a thorough method for comparing predictive accuracy. By focusing on the difference between the forecast errors from the two models, the DM test evaluates whether there is a statistically significant difference in their performance. For a more detailed description of the DM test see chapter 3.9.

Therefore, to validate the initial findings, there will be conducted a DM test. Which will provide a statistical backing for any claims regarding the relative performance of the OLS model versus the benchmark, ensuring that the conclusion is not only visually and intuitively appealing but also statistically sound. The results of this test will further inform us about the

consistency and reliability of the OLS model’s performance in forecasting energy consumption remainders after MSTL decomposition.


```R
 dm_test <- dm.test(tail(e1, n=8759), e2, alternative = "two.sided", h = 1, power = 2)

dm_test
```


    
    	Diebold-Mariano Test

    data:  tail(e1, n = 8759)e2
    DM = -24.566, Forecast horizon = 1, Loss function power = 2, p-value <
    2.2e-16
    alternative hypothesis: two.sided



Based on the results of the DM test comparing the errors of the OLS model and the benchmark model, the test statistic (DM = -24.566) and the extremely small p-value (p < 2.2e-16) strongly indicate a rejection of the null hypothesis at the 0.05 significance level. This implies that there is a significant difference between the forecasting errors of the two models.

In this context, the OLS model consistently underperforms compared to the benchmark model across the forecast horizon of 1, as reflected by the negative DM statistic. This result aligns with previous findings, supporting the idea that the benchmark model provides superior forecasts.

The outcome also reinforces the notion that the MSTL decomposition is highly effective in capturing the linear and seasonal structures within the time series. The remainders left after MSTL decomposition could be lacking linear structure, so that any attempt to predict them does not add significant value over a simple benchmark and might even be worse for forecasting the energy consumption than to ignore the remainder component.

### 5.6.3 Comparing Random forest with benchmark model

To understand whether the remainder component is truly unpredictable for the data, or if the relationship between the dependent and the independent variable are non-linear the results of the RF model will be compared the benchmark. For the RF model, it was earlier discovered that the optimal number of trees for the RF model is 250, the optimal training window size is 1 year, and that RF as well as OLS benefits from the highest possible training frequency, for this study the limit for the computational resources available is retraining every 1 timesteps


```R
# Create a data frame (table)
mse_table <- data.frame(
  Model = c("MSE_0", "MSE_RF"),
  MSE_Value = c(MSE_0, MSE_RF)
)

# Display the table
print(mse_table)
```

       Model MSE_Value
    1  MSE_0 33075.123
    2 MSE_RF  7374.672


From the table, the mean squared error (MSE) of the Random Forest (RF) model is significantly lower at 7,374.67 compared to the benchmark MSE of 33,075.12. This substantial reduction in error clearly indicates that the RF model is much more accurate in forecasting the remainder component than both the OLS model and the baseline approach of not forecasting at all.

The lower MSE values of the RF model suggest a more consistent performance across customer groups, reducing the overall prediction error and demonstrating robustness in forecasting the MSTL remainders.

Overall, the results highlight the Random Forest model as a more reliable method for predicting detrended and deseasonalized energy consumption. The significant reduction in median MSE reinforces its improvement over the baseline model,for good measure lets confirm the statistical significance of this improvement through a Diebold-Mariano test.


```R
 dm_test <- dm.test(e1, e3, alternative = "two.sided", h = 1, power = 2)

dm_test
```



    	Diebold-Mariano Test

    data:  e1e3
    DM = 60.43, Forecast horizon = 1, Loss function power = 2, p-value <
    2.2e-16
    alternative hypothesis: two.sided



The results of the Diebold-Mariano test further emphasize the superior accuracy of the Random Forest (RF) model over the benchmark. With a DM statistic of 60.43 and a p-value of less than 2.2e-16, the null hypothesis is strongly rejected, indicating a significant difference in the forecast errors between the RF model and the baseline approach.

These findings align with the earlier observations from the MSE distribution, reinforcing the conclusion that the RF model outperforms the approach of not forecasting at all. The RF model's ability to accurately predict detrended and deseasonalized energy consumption highlights the effectiveness of the MSTL decomposition in capturing the linear trend and seasonal components. By removing these elements, the RF model is able to focus on the remaining nonlinear patterns, offering valuable predictive insights that would otherwise be missed by simply predicting zero or relying on the benchmark.

This significant improvement in predictive performance suggests that the RF model is much better at capturing the complexities of the remainder component, further validating its use in this forecasting context.
