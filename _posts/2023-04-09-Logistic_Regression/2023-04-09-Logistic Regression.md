---
layout: post
title:  "Logistic Regression"
---

# Basics of Logistic Regression

The math expressions are broken in the github pages. Please see the markdown file in my github repository.
[CLICK here. md in Github repository](https://github.com/ubettercallsaul/ubettercallsaul.github.io/blob/master/_posts/2023-04-09-Logistic_Regression/2023-04-09-Logistic%20Regression.md)

Please let me know how to display Math expression in Github pages using Jekyll.


Logistic regression is one of the most popular classification methods. Its name contains 'regression', but it is popular as a classification method. Why? Let's visit the foundations of logistic regression with a simple example.

A logistic regression is used to to estimate the probability of an event. Suppose you have a dataset of independent variables and corresponding event occurance. For example, a dataset about the hours of studying as the independent variable and pass/fail for exam as the outcome event.

| **Hours** |  0.5 | 0.75 |   1  | 1.25 |  1.5 | 1.75 | 1.75 |   2  | 2.25 |  2.5 | 2.75 |   3  | 3.25 |  3.5 |   4  | 4.25 |  4.5 | 4.75 |   5  |  5.5 |
|:---------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|  **P/F**  | Fail | Fail | Fail | Fail | Fail | Fail | Pass | Fail | Pass | Fail | Pass | Fail | Pass | Fail | Pass | Pass | Pass | Pass | Pass | Pass |


```python
# Data
import numpy as np
Hours = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]).reshape(-1, 1)
Pass = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# plot figure
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(Hours, Pass, 'ok')
ax.set_xlabel('Hours of Studying')
ax.set_ylabel('Pass (1) /Fail (0) ')
ax.set_yticks([0, 1])
plt.show()
```


    
![image](https://user-images.githubusercontent.com/96639732/230798905-642da227-ab4e-4544-835f-6077372153b6.png)



Let's say you spent X-hours to prepare the exam. What is the probability to pass the exam? The logistic regression can be used to quantify the probability of discrete events at given independent variables.

## Logistic Distribution

The logistic regression starts from the logistic distribution. The CDF is described as

$$F(x,\mu,s) = {1\over 1+e^{-(x-\mu)/s}}$$

$\mu$ is the mean (location parameter) and *s* is a scale parameter where *s > 0*.
The CDF can be rewritten as the logistic function as

$$p(x) = {1\over 1+e^{-{(\beta_0+\beta_1 x)}}} $$

where $\beta_0=-\mu/s$ (intercept) and $\beta_1=1/s$ (rate parameter).
Why did we re-write the CDF of logistic distribution to the logistic function? Notice there is a familiar inear model, $\beta_0+\beta_1 x$. As you can probably tell, it is a huge benefit in regression modeling and its interpretation.


## Optimization
Now, everything is ready. The dataset (Hours of study and pass/fail result) and the logistic function to connect the dataset and the probability. So, the only unknown is the parameters of the logistic function. We should estimate the parameter values ($\beta{_0}$ and $\beta{_1}$) representing the dataset with the probability model the most by solving an optimization problem.

The cost function of the optimization problem can be defined with the sum of Log-loss function as below to measure the goodness of fit.

$$Loss(y_k, p_k) = -(y_k \ln (p_k) + (1 - y_k) \ln (1 - p_k))$$

The parameters can be obtained by minimizing the sum of the Log-loss values at the given dataset. This method is equivalent to the popular maximum likelihood estimation (MLE), which is to determine the parameters maximizing the joint likelihood.

$$Likelihood(y_k, p_k) = \prod_{k:y_k=1}p_k\,\prod_{k:y_k=0}(1-p_k)$$

As the optimization algorithm, BFGS is used as the default method in Scikit-learn's logistic regression.

## Logistic Regression with Scikit-learn


```python
# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

clf = LogisticRegression(random_state=0).fit(Hours, Pass)

print("Regression parameters")
print("  Intercept (beta0): {:.4f}".format(clf.intercept_[0]))
print("  Slope (beta1): {:.4f}".format(clf.coef_[0][0]))

score = clf.score(Hours, Pass)
print(f"Accuracy on Train Data: {score*100:.2f} %")
```

    Regression parameters
      Intercept (beta0): -3.1395
      Slope (beta1): 1.1486
    Accuracy on Train Data: 80.00 %
    


```python
# define logistic function
def LogisticFunction(x, b0, b1):
    p=1/(1+np.exp(-(b0+b1*x)))
    return p
```


```python
# Plot Data and Logistic Function
b0 = clf.intercept_[0]
b1 = clf.coef_[0][0]

x = np.arange(Hours.mean()-3*Hours.std(), Hours.mean()+3*Hours.std(), 0.1)
p = LogisticFunction(x, b0, b1)

predictions = clf.predict(Hours)
# find wrong predictions
idx_diff = [i for i, item in enumerate(predictions) if item!=Pass[i]]

# plot figure
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(Hours, Pass, 'ok')
ax.set_xlabel('Hours of studying')
ax.set_ylabel('Probability of Passing Eaxm')
ax.plot(x,p,'-b')
ax.set_title('Data and Logistic function', fontsize='10')
plt.grid()
plt.show()
```


    
![image](https://user-images.githubusercontent.com/96639732/230798918-a394daf7-62c7-4dc9-a36d-e71e4487afdf.png)
    



```python
proba = clf.predict_proba(Hours)
LogL = -metrics.log_loss(Pass, proba)
print("Log-Likelihood at the optimum: {:.4f}".format(LogL))
```

    Log-Likelihood at the optimum: -0.4109
    

At each test point, the logistic function returns the probability of each event (Pass/Fail). If $p(x)>0.5$, it is classified as 'Pass'. If not, 'Fail'.


```python
import pandas as pd

predictions = clf.predict(Hours)
# find predictions
idx_correct = [i for i, item in enumerate(predictions) if item==Pass[i]]
idx_wrong = [i for i, item in enumerate(predictions) if item!=Pass[i]]

summary = {'Hours': Hours.ravel(), 'Pass/Fail': Pass, 'Prediction':predictions ,'Probability of Fail':clf.predict_proba(Hours)[:,0], 'Probability of Pass':clf.predict_proba(Hours)[:,1]}
df = pd.DataFrame(summary)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Pass/Fail</th>
      <th>Prediction</th>
      <th>Probability of Fail</th>
      <th>Probability of Pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>0</td>
      <td>0</td>
      <td>0.928590</td>
      <td>0.071410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>0</td>
      <td>0</td>
      <td>0.907045</td>
      <td>0.092955</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.879840</td>
      <td>0.120160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.25</td>
      <td>0</td>
      <td>0</td>
      <td>0.846026</td>
      <td>0.153974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.50</td>
      <td>0</td>
      <td>0</td>
      <td>0.804808</td>
      <td>0.195192</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.75</td>
      <td>0</td>
      <td>0</td>
      <td>0.755741</td>
      <td>0.244259</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.75</td>
      <td>1</td>
      <td>0</td>
      <td>0.755741</td>
      <td>0.244259</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.698953</td>
      <td>0.301047</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.25</td>
      <td>1</td>
      <td>0</td>
      <td>0.635333</td>
      <td>0.364667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.50</td>
      <td>0</td>
      <td>0</td>
      <td>0.566605</td>
      <td>0.433395</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.75</td>
      <td>1</td>
      <td>1</td>
      <td>0.495216</td>
      <td>0.504784</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.00</td>
      <td>0</td>
      <td>1</td>
      <td>0.424021</td>
      <td>0.575979</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.25</td>
      <td>1</td>
      <td>1</td>
      <td>0.355846</td>
      <td>0.644154</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.50</td>
      <td>0</td>
      <td>1</td>
      <td>0.293056</td>
      <td>0.706944</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4.00</td>
      <td>1</td>
      <td>1</td>
      <td>0.189250</td>
      <td>0.810750</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4.25</td>
      <td>1</td>
      <td>1</td>
      <td>0.149054</td>
      <td>0.850946</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4.50</td>
      <td>1</td>
      <td>1</td>
      <td>0.116172</td>
      <td>0.883828</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4.75</td>
      <td>1</td>
      <td>1</td>
      <td>0.089778</td>
      <td>0.910222</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.00</td>
      <td>1</td>
      <td>1</td>
      <td>0.068914</td>
      <td>0.931086</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5.50</td>
      <td>1</td>
      <td>1</td>
      <td>0.040010</td>
      <td>0.959990</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot figure
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))
ax.set_xlabel('Hours of studying')
ax.set_ylabel('Probability of Passing Eaxm')
ax.plot(Hours[idx_correct],predictions[idx_correct], 'og')
ax.plot(Hours[idx_diff],predictions[idx_diff],'or')
ax.plot(x,p,'-b')
ax.legend(['Correct Prediction','Wrong Prediction','Logistic Function'], loc='lower right', fontsize='8')
ax.set_title('Prediction', fontsize='10')
plt.grid()
plt.show()
```


    
![image](https://user-images.githubusercontent.com/96639732/230798923-a6de0284-299b-4f10-8d2b-a4507af6edc4.png)
    


Confusion Matrix


```python
# Plot confusion matrix
import seaborn as sns
from matplotlib import pyplot as plt

cm = metrics.confusion_matrix(Pass, predictions)

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True)
plt.ylabel('Actual')
plt.xlabel('Predicted')
all_sample_title = 'Accuracy Score on Train Data: {:.2f}'.format(score)
plt.title(all_sample_title, size = 10)
plt.show()
```


    
![image](https://user-images.githubusercontent.com/96639732/230798927-637060ff-831b-4bb7-942c-8ee3aff9c77d.png)
    


There are two false positive and two false negative.

References
[Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution)

[Logit model](https://en.wikipedia.org/wiki/Logit)

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)

[Probit](https://en.wikipedia.org/wiki/Probit)

[Scikit-learn Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

[Scikit-learn Log-loss function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
