---
layout: post
title:  "Curve Fitting using SciPy and PyTorch"
---

# Curve Fitting using SciPy and PyTorch (Nonlinear Regression)

## Create a dataset


```python
# Define a quadratic function
c1 = 1
c2 = -4
c3 = 4

def func(x,c1,c2,c3):
  return c1*x**2 + c2*x + c3
```


```python
# Generate data. w/o noise
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


x = np.arange(10)
y = func(x,c1,c2,c3)

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![output1](https://user-images.githubusercontent.com/96639732/230232622-cf84be7e-e3ff-44ee-992f-d0be72b476fd.png)


## SciPy

Curve fitting


```python
# curve fitting
from scipy.optimize import curve_fit

popt, pcov = curve_fit(func, x, y)
y_pred1 = func(x, *popt)

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, y_pred1, 'r', label='SciPy curve_fit', alpha=0.8)
plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![output2](https://user-images.githubusercontent.com/96639732/230232668-ec13dd2e-b1d0-49dd-92ec-ebc7d280179d.png)
    


Parameters


```python
# Parameter estimates
from scipy.stats import t

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)

print(f"c1 (95%): {popt[0]:.2f} +/- {ts*np.diag(pcov)[0]:.2f}")
print(f"c2 (95%): {popt[1]:.2f} +/- {ts*np.diag(pcov)[1]:.2f}")
print(f"c3 (95%): {popt[2]:.2f} +/- {ts*np.diag(pcov)[2]:.2f}")

print(f"c1 error: {np.abs((c1-popt[0])/c1)*100:.2f} %")
print(f"c2 error: {np.abs((c2-popt[1])/c2)*100:.2f} %")
print(f"c3 error: {np.abs((c3-popt[2])/c3)*100:.2f} %")
```

    c1 (95%): 1.00 +/- 0.00
    c2 (95%): -4.00 +/- 0.00
    c3 (95%): 4.00 +/- 0.00
    c1 error: 0.00 %
    c2 error: 0.00 %
    c3 error: 0.00 %
    

## PyTorch

Define Regression class using Neural Network module that contains a linear function


```python
import torch
from torch import nn 
torch.manual_seed(1)

class RegressionModel(nn.Module):
  def __init__(self):
    super(RegressionModel, self).__init__()
    self.fc1 = nn.Linear(1, 30)
    self.fc2 = nn.Linear(30, 1)    

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

```


```python
model = RegressionModel()
print(model)
```

    RegressionModel(
      (fc1): Linear(in_features=1, out_features=30, bias=True)
      (fc2): Linear(in_features=30, out_features=1, bias=True)
    )
    

Initial point


```python
# Torch Tensor
xtt = torch.from_numpy(x.reshape(len(x),-1)).to(torch.float32)
ytt = torch.from_numpy(y.reshape(len(y),-1)).to(torch.float32)

# initial parameters
# c1_nn, c2_nn, c3_nn = model.parameters()
# print(c1_nn[0][0].item(), c2_nn[0].item(), c3_nn[0].item())

# print("Initial Parameters")
# print(f"slope : {a1:.2f}")
# print(f"intercept: {b1:.2f}")

plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, model(xtt).detach().numpy(), 'g', label='Torch NN (initial)', alpha=0.8)
plt.legend(loc='best')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()

```


    
![output3](https://user-images.githubusercontent.com/96639732/230232700-3c18bfa4-41c9-411e-b2ad-4ca07de13d38.png)
    


Training the model


```python
import torch.optim as optim
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0005)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
losses = []

for epocs in range(epochs):
  inputs = torch.from_numpy(x).float().unsqueeze(1)
  labels = torch.from_numpy(y).float().unsqueeze(1)

  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, labels)
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
plt.plot(x, model(xtt).detach().numpy(), 'g', label='Torch NN', alpha=0.8)
plt.legend(loc='lower right')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![output4](https://user-images.githubusercontent.com/96639732/230232722-9ef770a0-65db-40d4-b142-c308a7ec7443.png)
    



    
![output5](https://user-images.githubusercontent.com/96639732/230232753-be03c7ad-6497-4a72-9977-0c7bab88af3a.png)
    


## Comparison

Sum of Squared Error


```python
# Predictions at x
yfit1 = y_pred1
yfit2 = model(xtt).detach().numpy()

sse1 = np.sum((y-yfit1)**2)
print(f'SSE_SciPy = {sse1:.2e}')

sse2 = np.sum((y-yfit2.T)**2)
print(f'SSE_TorchNN = {sse2:.2e}')
```

    SSE_SciPy = 5.31e-29
    SSE_TorchNN = 4.61e+00
    


```python
plt.figure(figsize=(5,4))
plt.plot(x, y, 'bo', label='Data', alpha=0.8)
plt.plot(x, yfit1, 'r', label='SciPy Linear Regression', alpha=0.8)
plt.plot(x, yfit2, 'g', label='Torch NN', alpha=0.8)
plt.legend(loc='upper left')
plt.ylabel('y')
plt.xlabel('x')
plt.grid()
plt.show()
```


    
![output6](https://user-images.githubusercontent.com/96639732/230232778-b8d59a06-7302-49b9-a378-aa2e122b180a.png)
    


## Discussion

The NN model is not robust but too sensitive to the setting. In the study, NN is less accurate than the conventional regression model, but there are opportunities to improve the NN model as below.

1. Increase the number of hidden layers
2. Increase the number of neurons in each layer
3. Increase the number of epochs during training
4. Use a different activation function
5. Use a different optimizer



```python

```
