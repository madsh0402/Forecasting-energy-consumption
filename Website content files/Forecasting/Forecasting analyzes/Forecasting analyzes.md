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

# 5. Forecasting analyzes

Forecasting analysis is discussed in Chapter 5. It focuses on getting time series data ready for forecasting; the techniques involve min-max scaling and creating dummy variables, and lag \\( X \\) variables in order to forecast \\( R_t \\) into the future. After the techniques for data preparation has been used the MSTL decomposition is then used to find the remainders. The chapter also covers the length of the forecasting horizon, \\( h \\), and how it influences accuracy of predictions. The rolling window approach is also analyzed, the size of the static window and the frequency of retraining. Furthermore, the chapter also focuses on tuning parameters for Random Forest. After selecting the best prediction models for OLS and RF and then comparing them with benchmark models to analyze which model is proficient at forecasting the remainders.
