# PyTorch - Machine Learning from FreeCodeCamp

## PyTorch Workflow

1. Get Data ready by turning it into tensors
2. Build or pick a pretrained model
   - Pick a loss function & optimizer
   - Build a training loop
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload trained model

## Introduction to Tensors

scalar

- Integer
- Lower case variable

Vector

- Has a dimension of 1
- Has a shape of 2
- Lower case variable

Matrix

- Has a dimension of 2
- Has a shape of 2
- Matrix[1] first dimension gives the first brackets
- Upper case variable
- A matrix [3, 4] has 3 rows and 4 columns

TENSOR

- Has dimension of [1, 3, 3]
- This means the first dimension is a 3 x 3
- Upper case variable
- datatype is by default a float32

Random tensors

- Important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust the numbers to better represent the data

`Random numbers -> look at data -> update random numbers -> look at data -> update random numbers`

Image tensors

- [3, 244, 244] represents colour encoding, height and width
- 3 stands for red, green blue

Zeros and Ones

- You can create a tensor of all zeroes to mask an existing tensor. More common
- You can create a tensor of all ones

Range of tensors

- uses arange()

Tensors-like

- create a tensor full of zeroes that has the same shape as another tensor use zeros_like()

Tensor Datatypes

- default datatype for tensors even if you specify dtype as None, will always be float32. float32 and float16 are the most common
- `dtype` is what the datatype of the tensor is eg float16, float32
- `device` is what component tensor calculations will be done on
- `requires_grad` is if you want gradients of a tensor to be tracked

Precision

- How detailed something is represented
- float32 means a number has 32 bits = single precision
- float16 means a number has only 16 bits = half precision

Common errors

- Tensors not right datatype. Use `tensor.dtype`
- Tensors not right shape. Use `tensor.shape`
- Tensors not on the right device. Use `tensor.device`

Multiplying different datatypes

- Sometimes you will have errors and sometimes you wont

Manipulating tensors/operations

- Addition
- Subtraction
- Multiplication (element-wise)
- Division
- Matrix multiplication

Matrix multiplication (dot-product)

- Most common tensor operation
- Most common error will be shape mismatch
- To calculate, multiply the rows by the columns
- A 2x3 matrix dot product by a 3x2 will result in a 2x2 matrix
  - Inner dimensions must match
  - Resulting matrix has a shape of the outer dimensions

Matrix transpose

- To fix tensor shape issues, we can manipulate the shape of the tensors using a **transpose**
- Switches the axes or dimensions of a given tensor, for example

```
[7, 10],
[8, 11],
[9, 12]
```

will become

```
[7, 8, 9],
[10, 11, 12]
```

Min, max, mean, sum, etc

- torch.min, torch.max
- torch.mean but must be of float32 datatype to work
- torch.sum
- torch.argmin gives the index of the minimum
- torch.argmax gives the index of the maximum

Reshaping, stacking, squeezing, unsqueezing

- Reshaping = reshapes an input tensor to a defined shape
- View = returns a view of an input tensor of certain shape but keeps the same memory as the original tensor
  - If you change the view, you will also change the original vector
- Stacking = combine multiple tensors on top of eachother (hstack) or side by side (vstack)
- Squeeze = removes all `1` dimensions from a tensor
- Unsqueeze = add a `1` dimension to a target tensor
- Permute = rearranges the dimensions of a target tensor in a specified order. Returns a view

NumPy

- Scientific python numerical computing library
- Data is in NumPy but you usually want it into a PyTorch tensor
- `torch.from_numpy(ndarray)`
- `torch.Tensor.numpy()`
- Numpy default datatype is float64
- Tensor default datatype is float32
- When converting from numpy -> Pytorch, Pytorch will reflect numpy's default datatype of float64
- If you create a tensor from numpy you are create a new object in memory. Altering the original numpy data will not change new tensor

Reproducibility

- How a neural network learns is:
  `start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again`
- to reduce randomness in neural networks and PyTorch comes the concept of a **random seed**
- This is a 'flavour' of randomness
- In PyTorch use manual_seed(). Call it each time you create a random tensor
