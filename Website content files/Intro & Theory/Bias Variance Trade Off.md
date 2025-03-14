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

## 3.4 Bias variance trade-off

A fundamental aspect that affects a model's generalization ability is the tradeoff between bias and variance, two types of error that, together with irreducible error, constitute the expected prediction error of a model. Understanding and managing this tradeoff will help to create an accurate prediction model.

To understand the bias variance trade-off better, it is important to understand what bias and variance means. Bias in machine learning is the error introduced by approximating a real-world problem with a too-simplistic model. It arises from the assumptions made by the model to make the target function easier to learn. High bias can cause the model to miss the relevant relations between features and target outputs (underfitting), implying that the model is too simple to capture the underlying patterns in the data and therefore not making as accurate predictions. Variance, on the other hand, refers to the error from sensitivity to small fluctuations in the training set. A model with high variance pays a lot of attention to training data and does not generalize on the data which it has not been presented for beforehand. Essentially, it models the random noise in the training data, rather than the intended outputs (overfitting).

The bias-variance tradeoff highlights a key dilemma in machine learning: a model cannot simultaneously be more complex (to reduce bias) and simpler (to reduce variance) without compromise. A highly complex model may fit the training data very closely (low bias) but will likely fail to generalize well to unseen data (high variance). Conversely, a very simple model may generalize well because it is not sensitive to noise in the training data (low variance) but may fail to capture important patterns (high bias). Achieving a balance between bias and variance is therefore necessary to developing machine learning models that are both accurate and robust. This balance ensures that the model performs well not just on the training data but also on unseen data, which is the goal.

The complexity of a model has a major influence on both bias and variance. A more complex model, which is characterized by a larger number of parameters, has the capacity to reduce variance by closely fitting the training data. However, this can lead to an increase in complexity which corresponds to an increase in variance, as the model becomes too fitted to the training data, capturing noise as if it were signal, which has a negative impact on the model’s performance on new, unseen data.

Conversely, a simpler model, with fewer parameters or structures, may have higher variance as it might not be sophisticated enough to capture the underlying patterns in the data fully. Yet, it benefits from lower bias, making its predictions more reliable across different datasets and making its performance on unseen data better consequently. Therefore, the challenge lies in finding the optimal balance for the model’s complexity that achieves a middle ground between bias and variance, and thereby achieving, the best result.

The size of the training data set is a significant factor in managing bias and variance. Generally, increasing the amount of training data can reduce the model's variance without necessarily increasing bias. This reduction occurs because more data provide a better approximation of the real-world distribution, helping the model to generalize better and be less sensitive to the noise within any sample of training data.

However, it's worth noting that simply adding more data is not improvement, especially if the added data is noisy or if the model has reached some limit beyond which it cannot learn from additional data. Adding additional data could also change the focus towards other methods of improving the model. (Luxburg et al, 2011 and Hastie et al, 2017).
