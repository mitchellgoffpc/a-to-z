import torch
import unittest
import numpy as np
from helpers import Tensor
from ops.reduce_ops import Sum, Mean, Min, Max
from parameterized import parameterized

np.random.seed(42)

class TestReduceOps(unittest.TestCase):
  @parameterized.expand([(Sum, torch.sum), (Mean, torch.mean), (Min, torch.min), (Max, torch.max)])
  def test_unary_op(self, my_op, torch_op):
    data = np.random.normal(size=(16, 16)).astype(np.float32)

    # Test forward
    my_data = Tensor(data)
    my_result = my_op.apply(my_data)
    torch_data = torch.tensor(data, requires_grad=True)
    torch_result = torch_op(torch_data)
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy(), atol=1e-5)

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_data.grad.numpy(), torch_data.grad.numpy())
