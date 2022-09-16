import torch
import unittest
import numpy as np
from helpers import Tensor
from unary_ops import Negate, Relu
from parameterized import parameterized

class TestUnaryOps(unittest.TestCase):
  @parameterized.expand([(Negate, torch.neg), (Relu, torch.nn.functional.relu)])
  def test_unary_op(self, my_op, torch_op):
    data = np.random.normal(size=(32, 32)).astype(np.float32)

    # Test forward
    my_data = Tensor(data)
    my_result = my_op.apply(my_data)
    torch_data = torch.tensor(data, requires_grad=True)
    torch_result = torch_op(torch_data)
    np.testing.assert_equal(my_result.numpy(), torch_result.detach().numpy())

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_equal(my_data.grad.numpy(), torch_data.grad.numpy())
