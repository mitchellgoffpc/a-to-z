import torch
import unittest
import numpy as np
from relu import Relu
from helpers import Tensor

class TestRelu(unittest.TestCase):
  def test_relu_forward(self):
    data = np.random.normal(size=(32, 32)).astype(np.float32)
    my_input = Tensor(data)
    my_result = Relu.apply(my_input)
    torch_input = torch.tensor(data, requires_grad=True)
    torch_result = torch.nn.functional.relu(torch_input)
    np.testing.assert_equal(my_result.numpy(), torch_result.detach().numpy())

    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_equal(my_input.grad.numpy(), my_input.grad.numpy())
