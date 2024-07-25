---
---
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

## 3.5 Hyperparameter tuning

Hyperparameters are parameters where the values are determined before the learning process begins, unlike model parameters, which are derived through training. Hyperparameters will influence the learning process and the resulting model's performance. Examples of hyperparameters include the learning rate or the number of trees in a Random Forest (RF) model.

Tuning hyperparameters is essential because they will control the behavior of the training algorithm directly and thereby affect the performance of the model. Optimal hyperparameter settings can improve model accuracy, reduce overfitting, and ensure that the model generalizes well from the training data to unseen data. Thus, hyperparameter tuning is an important component of machine learning. Hyperparameter tuning involves selecting a combination of hyperparameters that delivers the best performance. Common techniques for hyperparameter tuning include grid search and random search.

One technique is grid search, which involves defining a grid of hyperparameter values and evaluating model performance for each combination. This method can be computationally expensive. Another well-known method is random search, which unlike grid search randomly selects combinations of hyperparameters to evaluate. This method is often more efficient than grid search, as it does not systematically explore every combination but instead samples them at random.

The RF algorithm is particularly sensitive to several hyperparameters that can significantly influence both the training efficiency and the predictive power of the model. Notable hyperparameters in RF include:
- **Number of Trees:** The number of trees in the forest. Increasing the number of trees can lead to better performance but also increases computational cost and training time.
- **Mtry:** Specifies the number of features to consider for the best split at each tree node. Adjusting 'mtry' balances model accuracy and overfitting.
- **Depth of Trees:** The maximum depth of each tree. Deeper trees can model more complex patterns but can also lead to overfitting.
- **Minimum Samples Split:** The minimum number of samples required to split an internal node. Higher values prevent the model from learning overly specific patterns, thus reducing overfitting.
- **Minimum Samples Leaf:** The minimum number of samples required to be at a leaf node. Setting this parameter can provide a means of control against overfitting similar to the minimum sample split.

Each of these hyperparameters can impact the RF model in different ways. For example, increasing the number of trees generally improves model performance due to the reduction in variance from averaging multiple trees, though at a cost of increased computational expense. Similarly, deeper trees allow the model to learn more detailed data specifics, enhancing training accuracy, but might decrease generalization to new data unless carefully controlled. The 'mtry' setting directly influences model complexity and robustness by dictating the number of features considered at each split, significantly affecting both model bias and variance. Finally, the settings of minimum samples split and minimum samples leaf influence how fine the decision boundaries can be, impacting both the bias and variance of the model. Properly tuning these hyperparameters is therefore essential to achieve the perfect balance in the trade-off between underfitting and overfitting, thereby optimizing the forecasting model's performance. (Hyperparameters and Tuning Strategies for Random Forest, n.d.)

The bias-variance trade-off, as stated in chapter 3.4, is a fundamental concept in machine learning that describes the tension between bias and variance. In the context of RF, specific hyperparameters directly influence this trade-off. Shallow trees (indicated by low depth) or too few trees can result in too high of a bias, as the model may not capture enough of the complexity in the data. At the same time, high variance can be a result of overly complex models, such as having too many deep trees, which fit excessively to the outliers in the training data. Adjusting hyperparameters such as the number of trees, the depth of trees, minimum samples split, and minimum samples leaf affects how well the model balances these errors. Typically, more trees can reduce variance without significantly increasing bias, whereas increasing tree depth can decrease bias but increase variance.

Achieving an optimal balance between bias and variance involves adjusting hyperparameters to fine-tune the model's capacity to generalize to new data. One of the strategies for tuning hyperparameters in RF is the Ensemble method. This method is used to increase the number of trees in the forest, which often helps in reducing variance, as the averaging of multiple trees tends to cancel out individual errors. Although this method can reduce the variance, the trade-off between bias and variance should be considered. Therefore, the adjustment of the number of trees is important. The goal is to have both bias and variance at acceptable levels, resulting in a model with accurate predictive power on new, unseen data. (Hyperparameters and Tuning Strategies for Random Forest, n.d.).

In Chapter 5 of this thesis, the application of hyperparameter tuning in the context of predicting detrended and deseasonalized energy consumption following an MSTL decomposition will be explored. Hyperparameter tuning is crucial because it balances computational efficiency with predictive accuracy, a key consideration given the complexity and size of time series data in energy consumption.
