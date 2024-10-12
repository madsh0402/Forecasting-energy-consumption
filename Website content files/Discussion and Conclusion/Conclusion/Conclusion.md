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

# 7. Conclusion

In this thesis, there has been explored machine learning models applied to decomposed energy consumption time series, focusing on the use of MSTL. This thesis was initiated in collaboration with Andel Energi who wanted to see machine learning applied to their forecasting objectives, in the light is the recent energy crises resulting in a shift in consumer behavior.

The central research question posed at the beginning of this study was:

1. Can machine learning models be used to forecast MSTL decomposed energy consumption?

The findings presented in this thesis proved that machine learning techniques, specifically RF models, can effectively forecast the remainders of MSTL decomposed energy consumption time series. This ability to forecast what was thought to be unpredictable or random noise within the data shows the importance of this study since it could be proven essential in future energy consumption predictions.

Throughout the analysis, it was found that the OLS model performs less effectively compared to a more complex model like RF. This inaccuracy in forecasting stems from the OLS model’s linear nature, which limits its ability to capture and learn from the nonlinear relationships present in the detrended and deseasonalized energy consumption. Consequently, OLS was even outperformed by a naive model, which simply assumed that there were no remainders after the detrending and deserialization.

Chapter 5 provided an analysis of the forecasting framework of the detrended and deserialized energy consumption. It was observed that factors such as retraining frequency and window size critically influences the accuracy of the forecast. A balanced approach that considers both historical data and the responsiveness to new input, meaning a training window size of one year was favorable. The results showed that RF models was capable of predicting the remainders of the MSTL decomposition, opening up to the idea of further improvement via the use of Boosting.

This thesis leaves several pathways for future research. Further exploration into additional machine learning models that could offer even greater improvements in forecasting accuracy. Moreover, a more detailed analysis of the other decomposed components of the energy consumption time series could be interesting as this would be needed to estimate the full energy consumption. Creating a framework of using different forecasting models for different MSTL components.
