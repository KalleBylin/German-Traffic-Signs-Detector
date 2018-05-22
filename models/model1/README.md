## Logistic regression

**What is it?**

Logistic regression is a classification algorithm often used for predictive analysis and tries to explain the relationship between one dependent binary variable and one or more independent variables. A short explanation posted on Analytics Vidhya is that "it predicts the probability of ocurrence of an event by fitting data to a logit function".

A real-world example for logistic regression could be as a predictor for wether a patient has a disease, like diabetes, or not, based on certain characteristics of the patient like age, body mass index, sex, results tests, etc.

**Strengths of the model**

Logistic regression does not make some of the key assumptions of linear regression and general linear models in terms of, for example, linearity, normality, homoscedasticity, and measurement level. It works with relationships that are not linear because it applies a non-linear log transformation to the predicted odds ratio. It is also popular because its results are relatively easy to interpret. The estimated regression coefficients can be back-transformed off of a log scale to interpret the conditional effects of each feature.

**Weaknesses of the model**

Still, as every model, it has its trade-offs and weaknesses. Logistic regression is a classification model for discrete, binary problems. In order to predecit multiple classes a scheme like one-vs-rest (OvR) has to be used. High correlation variables and outliers can make the model perform poorly. It sometimes needs large datasets because maximum likelihood estimates are less powerful than the ordinary least squares used for linear regression.

(Sources: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html, http://www.statisticssolutions.com/what-is-logistic-regression/, http://www.statisticssolutions.com/what-is-logistic-regression, http://www.statisticssolutions.com/assumptions-of-logistic-regression/)