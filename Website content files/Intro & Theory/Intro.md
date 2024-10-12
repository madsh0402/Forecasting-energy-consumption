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

# Intro

This thesis explores the application of machine learning techniques and time series decomposition to forecast energy consumption in Denmark. It particularly focuses on the Multi-Seasonal-Trend Decomposition using Loess (MSTL) to isolate underlying seasonal and trend components within the energy data, allowing forecasting of the detrended and deseasonalized energy consumption. The study employs machine learning models, specifically Ordinary Least Squares (OLS) and Random Forest (RF), to model the remainders derived from MSTL decomposition. This approach improves the predictive accuracy of the forecasts by seperating consumption patterns influenced by varying seasonal impacts and external variables such as weather conditions and economic factors.
The findings demonstrate that using MSTL for decomposition paired with machine learning algorithms can effectively forecast detrended and deseasonalized energy consumption. The thesis contributes by detailing the performance comparisons of these models and discussing the practical implications for improving energy forecasting.
The study shows that the RF model outperforms OLS and a baseline model, the study does however discuss that RF comes with a significant computational requirement that could limit the retraining frequency aswell as the number of trees used. These reuslts additionally shows that MSTL is efficient in capturing the linearity in the energy consumption as trend and seasonal components. Therefor the study proves that it is possible and beneficial to forecast the remainder component of the energy consumption, but also recognise that to estimate the full energy consumtion the trend and seasonal components would have to be forecasted aswell.
