
# Types of Activation Functions in Neural Networks

## Introduction

Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns. They determine the output of a neural network node and the decision boundaries it can learn. Without activation functions, the neural network would behave like a linear regression model, regardless of its depth.

## Types of Activation Functions

### Sigmoid

The sigmoid function is an S-shaped curve that maps any input to a value between 0 and 1.

**Mathematical Formula:**
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

**Pros:**
- Smooth gradient, preventing abrupt changes.
- Output values bound between 0 and 1, useful for probability-based outputs.

**Cons:**
- Vanishing gradient problem: gradients become very small for large positive or negative inputs, slowing down training.
- Outputs are not zero-centered, which can slow down convergence.

**Applications:**
- Mainly used in binary classification problems.
- Often used in the output layer of a binary classifier.

**Python Implementation:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Derivative:**
\[ \sigma'(z) = \sigma(z)(1 - \sigma(z)) \]

### Tanh

The tanh function is similar to the sigmoid but outputs values between -1 and 1.

**Mathematical Formula:**
\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]

**Pros:**
- Outputs are zero-centered, which helps in faster convergence.
- Smooth gradient, preventing abrupt changes.

**Cons:**
- Vanishing gradient problem similar to the sigmoid function.

**Applications:**
- Commonly used in hidden layers of neural networks.
- Preferred over sigmoid in practice due to zero-centered output.

**Python Implementation:**

```python
def tanh(z):
    return np.tanh(z)
```

**Derivative:**
\[ \tanh'(z) = 1 - \tanh^2(z) \]

### ReLU (Rectified Linear Unit)

ReLU is the most commonly used activation function in deep learning.

**Mathematical Formula:**
\[ \text{ReLU}(z) = \max(0, z) \]

**Pros:**
- Solves the vanishing gradient problem.
- Computationally efficient, requiring only a threshold at zero.

**Cons:**
- Dying ReLU problem: neurons can stop learning entirely if they output zero for all inputs.

**Applications:**
- Widely used in hidden layers of neural networks.
- Preferred in convolutional neural networks (CNNs) and deep learning models.

**Python Implementation:**

```python
def relu(z):
    return np.maximum(0, z)
```

**Derivative:**
\[ \text{ReLU}'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases} \]

### Leaky ReLU

Leaky ReLU is a variation of ReLU that allows a small, non-zero gradient when the input is negative.

**Mathematical Formula:**
\[ \text{Leaky ReLU}(z) = \begin{cases} 
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0 
\end{cases} \]

where \(\alpha\) is a small constant, typically 0.01.

**Pros:**
- Prevents the dying ReLU problem.

**Cons:**
- Introduces a small gradient for negative inputs, which can still slow down learning.

**Applications:**
- Used in hidden layers to avoid the dying ReLU problem.
- Suitable for models where ReLU is not performing well due to dying neurons.

**Python Implementation:**

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
```

**Derivative:**
\[ \text{Leaky ReLU}'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
بالتأكيد، سأقوم بإنشاء ملف README جديد يحتوي على شرح مفصل لأنواع دوال التفعيل في الشبكات العصبية. سأبدأ من البداية وأضيف الشرح المفصل بدون استخدام الصور.

### ملف `activation_functions.md`

```markdown
# Types of Activation Functions in Neural Networks

## Introduction

Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns. They determine the output of a neural network node and the decision boundaries it can learn. Without activation functions, the neural network would behave like a linear regression model, regardless of its depth.

## Types of Activation Functions

### Sigmoid

The sigmoid function is an S-shaped curve that maps any input to a value between 0 and 1.

**Mathematical Formula:**
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

**Pros:**
- Smooth gradient, preventing abrupt changes.
- Output values bound between 0 and 1, useful for probability-based outputs.

**Cons:**
- Vanishing gradient problem: gradients become very small for large positive or negative inputs, slowing down training.
- Outputs are not zero-centered, which can slow down convergence.

**Applications:**
- Mainly used in binary classification problems.
- Often used in the output layer of a binary classifier.

**Python Implementation:**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Derivative:**
\[ \sigma'(z) = \sigma(z)(1 - \sigma(z)) \]

### Tanh

The tanh function is similar to the sigmoid but outputs values between -1 and 1.

**Mathematical Formula:**
\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]

**Pros:**
- Outputs are zero-centered, which helps in faster convergence.
- Smooth gradient, preventing abrupt changes.

**Cons:**
- Vanishing gradient problem similar to the sigmoid function.

**Applications:**
- Commonly used in hidden layers of neural networks.
- Preferred over sigmoid in practice due to zero-centered output.

**Python Implementation:**

```python
def tanh(z):
    return np.tanh(z)
```

**Derivative:**
\[ \tanh'(z) = 1 - \tanh^2(z) \]

### ReLU (Rectified Linear Unit)

ReLU is the most commonly used activation function in deep learning.

**Mathematical Formula:**
\[ \text{ReLU}(z) = \max(0, z) \]

**Pros:**
- Solves the vanishing gradient problem.
- Computationally efficient, requiring only a threshold at zero.

**Cons:**
- Dying ReLU problem: neurons can stop learning entirely if they output zero for all inputs.

**Applications:**
- Widely used in hidden layers of neural networks.
- Preferred in convolutional neural networks (CNNs) and deep learning models.

**Python Implementation:**

```python
def relu(z):
    return np.maximum(0, z)
```

**Derivative:**
\[ \text{ReLU}'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0 
\end{cases} \]

### Leaky ReLU

Leaky ReLU is a variation of ReLU that allows a small, non-zero gradient when the input is negative.

**Mathematical Formula:**
\[ \text{Leaky ReLU}(z) = \begin{cases} 
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0 
\end{cases} \]

where \(\alpha\) is a small constant, typically 0.01.

**Pros:**
- Prevents the dying ReLU problem.

**Cons:**
- Introduces a small gradient for negative inputs, which can still slow down learning.

**Applications:**
- Used in hidden layers to avoid the dying ReLU problem.
- Suitable for models where ReLU is not performing well due to dying neurons.

**Python Implementation:**

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
```

**Derivative:**
\[ \text{Leaky ReLU}'(z) = \begin{cases} 
1 & \text{if } z > 0 \\
\alpha & \text{if } z \leq 0 
\end{cases} \]

### Softmax

The softmax function is used primarily in the output layer of neural networks for classification tasks with multiple classes.

**Mathematical Formula:**
\[ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \]

**Pros:**
- Converts logits into probabilities that sum to 1.

**Cons:**
- Computationally expensive due to the exponentiation operation.

**Applications:**
- Used in the output layer of neural networks for multi-class classification problems.
- Converts the output layer's logits to a probability distribution.

**Python Implementation:**

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)
```

**Derivative:**
\[ \frac{\partial \text{Softmax}(z_i)}{\partial z_j} = \text{Softmax}(z_i) (\delta_{ij} - \text{Softmax}(z_j)) \]

## Comparison of Activation Functions

| Function | Range | Derivative | Pros | Cons |
|----------|-------|------------|------|------|
| Sigmoid  | (0, 1)| \(\sigma(z)(1 - \sigma(z))\) | Smooth gradient, probability-based output | Vanishing gradient, not zero-centered |
| Tanh     | (-1, 1)| \(1 - \tanh^2(z)\) | Zero-centered, smooth gradient | Vanishing gradient |
| ReLU     | [0, ∞) | 1 (z > 0), 0 (z ≤ 0) | Solves vanishing gradient, efficient | Dying ReLU problem |
| Leaky ReLU | (-∞, ∞) | 1 (z > 0), \(\alpha\) (z ≤ 0) | Prevents dying ReLU | Small gradient for negatives |
| Softmax  | (0, 1)| Complex | Probability distribution output | Computationally expensive |

## Conclusion

Choosing the right activation function is crucial for the performance of a neural network. Each function has its strengths and weaknesses, and the choice depends on the specific problem and network architecture. Understanding these functions helps in building more efficient and accurate neural networks.

## References

- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)

