## 5.8 Summary

In this chapter, the forecasting of detrended and deseasonalized energy consumption has been analyzed and has provided an insight into the framework that have been taken into use.

The assessment of the forecasting horizon’s impact was found to not have a large impact on the forecasting accuracy, meaning that the framework is useable to forecast into the future by adjusting the forecasting horizon, \\( h \\), ensuring the framework ability to forecast long range energy consumption.

Moreover, the analysis of the effect of the rolling window approach showed that retraining frequency had a large impact on the accuracy of the forecast. The findings showed that the accuracy of predictions is improved by frequent model retraining, however, this has a high impact on the amount of computational resources, meaning that it would be needed to balance computational resources with the optimal forecast precision, so the retraining frequency should be as high as the computational resources available allows.
The analysis also looked at the window size used for training which also showed to have considerable influence on the model performance. The analysis showed that a one-year window size emerged as an effective balance, capturing enough historical data to account for hidden seasonality and trend while retaining responsiveness to new data for both OLS and RF.

When it came to the evaluation of forecast results, the comparison between the OLS and RF models highlighted RF as the more accurate forecast. This stems from the RF model’s ability to discern and learn from the nonlinear relationships between independent variables and the MSTL remainders.

The analysis concluded that prediction of the remainders after MSTL decomposition is indeed viable, further forecasting of trend and seasonal components is necessary to provide an estimation of the energy consumption. The results show that although the remainders are less predictable than the trend and seasonal components due to their nature, they are not entirely random and can be modeled to improve forecast accuracy.

To summarize, chapter 5 has provided the understanding of the choice of model, retraining frequency, and training window size which all play an important role in the accuracy of forecasting the detrended and deseasonalized energy consumption, while the forecasting horizon can be freely chosen to lineup with the objective. Moreover, it offers an idea for how forecasting of energy consumption can be done and for future research to build upon. Particularly exploring additional models that may further improve forecasting accuracy for MSTL decomposed time series, as well as demonstrate the predictability of the other components.
