---
layout: post
title:  "Linear Regression using SciPy and PyTorch"
---

# Linear Regression using SciPy and PyTorch

## Create a dataset


```python
a = 2
b = 1
def func(x,a,b):
  return a * x + b
```


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

x = np.arange(10)
y = func(x,a,b)

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![fig1](https://github.com/ubettercallsaul/ubettercallsaul.github.io/blob/master/_posts/2023-04-01-Linear_Regression_using_SciPy_and_Torch/2023-04-01-Linear_Regression_using_SciPy_and_Torch_3_0.png?raw=true)
    


## SciPy

Linear Regression


```python
from scipy import stats
res = stats.linregress(x, y)

print(f"R-squared: {res.rvalue**2:.6f}")

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, res.intercept + res.slope*x, 'r', label='SciPy Linear Regression', alpha=0.8)
plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()

```

    R-squared: 1.000000
    


    
![image](https://user-images.githubusercontent.com/96639732/229319633-b0055815-90bd-4cf0-835d-f15590acc709.png)

    


Parameters


```python
from scipy.stats import t

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)

print(f"slope (95%): {res.slope:.2f} +/- {ts*res.stderr:.2f}")
print(f"intercept (95%): {res.intercept:.2f}"f" +/- {ts*res.intercept_stderr:.2f}")

print(f"slope error: {np.abs(a-res.slope)/a*100:.2f} %")
print(f"intercept error: {np.abs(b-res.intercept)/b*100:.2f} %")
```

    slope (95%): 2.00 +/- 0.00
    intercept (95%): 1.00 +/- 0.00
    slope error: 0.00 %
    intercept error: 0.00 %
    

## PyTorch


```python
import torch
from torch import nn 
torch.manual_seed(1)

```




    <torch._C.Generator at 0x1f2dcf11110>



Define Linear Regression class using Neural Network module that contains a linear function


```python
class LinearRegressionModel(nn.Module):
  def __init__(self):
    super(LinearRegressionModel, self).__init__()
    self.linear = nn.Linear(1, 1)

  def forward(self, x):
    pred = self.linear(x)
    return pred
```

Initial point


```python
model = LinearRegressionModel()

print(model)
print(list(model.parameters()))
```

    LinearRegressionModel(
      (linear): Linear(in_features=1, out_features=1, bias=True)
    )
    [Parameter containing:
    tensor([[0.5153]], requires_grad=True), Parameter containing:
    tensor([-0.4414], requires_grad=True)]
    







```python
# Torch Tensor
xtt = torch.from_numpy(x.reshape(len(x),-1)).to(torch.float32)
ytt = torch.from_numpy(y.reshape(len(y),-1)).to(torch.float32)

# initial parameters
a_nn, b_nn = model.parameters()
a1, b1 = a_nn[0][0].item(), b_nn[0].item()

print("Initial Parameters")
print(f"slope : {a1:.2f}")
print(f"intercept: {b1:.2f}")

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, model(xtt).detach().numpy(), 'y', label='Torch NN (initial)', alpha=0.8)
plt.legend(loc='best')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()

```

    Initial Parameters
    slope : 0.52
    intercept: -0.44
    

![image](https://user-images.githubusercontent.com/96639732/229319650-93bf76fc-d732-4504-a04d-718bb218bb18.png)

    

    


Training the model


```python
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


epochs = 500

losses = []

for epocs in range(epochs):
  optimizer.zero_grad()

  y_pred = model(xtt)
  loss = criterion(y_pred, ytt)
  losses.append(loss.item())
  loss.backward()

  optimizer.step()

plt.figure(figsize=(5,4))
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, model(xtt).detach().numpy(), 'y', label='Torch NN', alpha=0.8)
plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    

    



    
![image](https://user-images.githubusercontent.com/96639732/229319656-f466ac01-cf38-44d3-a7a0-2d412fe9515d.png)

![image](https://user-images.githubusercontent.com/96639732/229319666-b119d0e0-984a-4554-9396-149669544c4b.png)

    



```python
a2, b2 = a_nn[0][0].item(), b_nn[0].item()

print("Parameters")
print(f"slope : {a2:.2f}")
print(f"intercept: {b2:.2f}")

print(f"slope error: {np.abs(a-a2)/a*100:.2f} %")
print(f"intercept error: {np.abs(b-b2)/b*100:.2f} %")
```

    Parameters
    slope : 2.01
    intercept: 0.93
    slope error: 0.55 %
    intercept error: 6.92 %
    

## Comparison

Sum of Squared Error


```python
# Predictions at x
yfit1 = res.intercept + res.slope*x
yfit2 = model(xtt).detach().numpy()

sse1 = np.sum((y-yfit1)**2)
print('SSE_SciPy = ', sse1)

sse2 = np.sum((y-yfit2.T)**2)
print('SSE_TorchNN = ', sse2)

```

    SSE_SciPy =  0.0
    SSE_TorchNN =  0.013869207169875608
    


```python
plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, yfit1, 'r', label='SciPy Linear Regression', linewidth=3, alpha=0.8)
plt.plot(x, yfit2, 'y', label='Torch NN', alpha=1)
plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![image](https://user-images.githubusercontent.com/96639732/229319670-cbcf96b0-1cad-425a-a1e5-534c8e1c7c9a.png)

    

