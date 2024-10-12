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

## 3.6 Machine learning in time series

A known challenge in linear regression is the risk of overfitting, especially as the number of predictors rises. In such situations, the linear model may begin to capture noise instead of meaningful insights. The OLS method is made so it minimizes the sum of squared errors for the training data. This approach leads to models that are unbiased in terms of the training data, however, it might exhibit high variance (Hastie et al., 2017). Additionally, when dealing with numerous parameters, understanding, and interpreting each variable's contribution becomes difficult, and in some cases, misleading. It's often more advantageous to select a subset of variables that have the most substantial predictive power. 

Machine learning involves learning from past experiences to perform a task, with the aim that this learned capability can be generalized to future, similar situations. This field encompasses a wide range of models, each with its unique set of strengths and weaknesses. When applying these models to time series data, such as energy data, careful consideration and an understanding of the models' assumptions and limitations are important. Typically, machine learning models presume that observations are independent and identically distributed iid.. However, this assumption often does not apply for energy series data, where there is a notable autocorrelation of absolute consumption over extended periods. Overlooking the learning behaviors of machine learning algorithms may result in considerable challenges when trying to model time series data.

For training and validating models to ensure the least amount of bias while tuning model parameters to get the lowest possible error Cross-validation is used. This serves a dual purpose: firstly, to estimate the model's performance on unseen data, thereby helping to avoid overfitting; and secondly, to aid in the selection of optimal hyperparameters for the algorithm. Traditional cross-validation techniques split the dataset into K segments of roughly equal size. For each segment (the kth part), the model is trained on the remaining \\(K-1\\) segments and then tested on the kth segment to evaluate prediction errors. This process is repeated across all \\(K\\) segments. The prediction error is quantified using a loss function, \\(\ell (y_i,\hat y_i)\\), which measures the cost of predicting \\(\hat y\\) when the true value is \\(y_i\\). While the least square error:

$$ \ell (y_i,\hat y_i) = ||y_i-\hat y_i ||^2 $$

 is commonly used in regression problems, the choice of loss function can vary depending on the specific goals of the modeling task. However, the traditional approach to cross-validation, which involves random partitioning of the dataset, may not be suitable for time series data due to the temporal dependencies among observations. This aspect requires adaptation or modification of the cross-validation technique to account for the sequential nature of time series data, ensuring that the validation process remains meaningful and avoids introducing lookahead bias.

### 3.6.1 Evaluation metrics

For assessing the efficacy of various machine learning algorithms, an essential factor is the use of an evaluation metric. This metric is a measure of a model's performance, closely aligned with, and often the same as, the loss function used during training. Essentially, this metric quantifies the accuracy of predictions made on unseen data,  and then translates the outcome into a numerical value that represents the error cost. While the loss function is integral to the training phase, the Mean Squared Error (*MSE*) is frequently chosen for both model estimation and as the benchmark for evaluating performance out-of-sample. It is best to have various evaluation matrices when evaluating the performance of models. Two of these evaluation matrices that are commonly used is Root Mean Squared Error (*RMSE*) and Mean Absolute Error (*MAE*), which are variations of MSE but with distinct calculations (Hodson, 2022).

*RMSE* Is defined as:

$$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2} $$

This calculation ensures that the metric’s unit match those of the target variable, which helps in the interpretability within the dataset’s context.

*MAE* is calculated as:

$$ MAE = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \hat{y}_i \right| $$

*MAE* is preferable when assessing error magnitude without improperly penalizing outliers.

In situations where forecasting precision is paramount, and larger outliers carry significant costs, like energy consumption or other financial datasets, *MSE* and *RMSE* are preferred for their stringent penalization of such outliers, as supported by Hastie et al., 2017. This approach is consistent with the common machine learning principle of aligning the evaluation metric with the loss function.
