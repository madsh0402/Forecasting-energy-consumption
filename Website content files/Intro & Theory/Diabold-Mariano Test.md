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

## 3.9 Diebold-Mariano Test

The DM test, introduced by Diebold and Mariano in 1995, is a basis for statistically comparing the predictive accuracy of two forecasting models. The essence of the DM test lies in its null hypothesis, which posits that both models under comparison have equivalent forecasting accuracy. The DM-test statistic is derived from:

$$ DM = \frac{\bar{d}}{\sqrt{\text{Var}(\bar{d})}} $$

Here, \\( \bar{d} \\) represents the mean of differences in squared forecast error between the two models, which can be written as:

$$ \bar{d} = \frac{1}{n} \sum_{t=1}^n \left( \hat{e}_{i,t}^2 - \hat{e}_{j,t}^2 \right) $$

And \\(\text{Var}(\bar{d})\\) is the variance of these differences. A statistically significant \\( DM \\) would indicate a visible difference in the predictive accuracies of the two models being compared. This can be used to determine which model is superior as a forecasting tool (Diebold et al., 1995).

There is a modified DM Test called the Harvey, Leybourne, & Newbold (HLN) Test. This test expands the DM test by adjusting for size distortions and enhancing reliability, particularly under scenarios where forecasting errors may not have normally distributed patterns. The HLN test statistic is formulated to capture the differences in the mean squared errors (MSEs), adjusted for potential heteroskedasticity (changes in variance across the sample) and autocorrelation (correlation of a variable with its past values) in the forecast errors (Harvey et al., 1997).



```python

```
