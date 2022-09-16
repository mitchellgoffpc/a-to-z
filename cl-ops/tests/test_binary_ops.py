import torch
import unittest
import numpy as np
from helpers import Tensor
from binary_ops import Add, Sub, Mul, Div, Pow
from parameterized import parameterized

np.random.seed(42)

class TestBinaryOps(unittest.TestCase):
  @parameterized.expand([
    (Add, torch.add),
    (Sub, torch.sub),
    (Mul, torch.mul),
    (Div, torch.div, 0, 0, 1e-5),
    (Pow, torch.pow, 0, 1e-7, 1e-7)])

  def test_binary_op(self, my_op, torch_op, tol_out=0, tol_a=0, tol_b=0):
    a = np.random.normal(size=(32, 32)).astype(np.float32)
    b = np.random.normal(size=(32, 32)).astype(np.float32)

    # Test forward
    my_a, my_b = Tensor(a), Tensor(b)
    my_result = my_op.apply(my_a, my_b)
    torch_a = torch.tensor(a, requires_grad=True)
    torch_b = torch.tensor(b, requires_grad=True)
    torch_result = torch_op(torch_a, torch_b)
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy(), atol=tol_out)

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_a.grad.numpy(), torch_a.grad.numpy(), atol=tol_a)
    np.testing.assert_allclose(my_b.grad.numpy(), torch_b.grad.numpy(), atol=tol_b)
