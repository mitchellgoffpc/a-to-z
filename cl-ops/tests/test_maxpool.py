import torch
import unittest
import numpy as np
from helpers import Tensor
from ops.maxpool import MaxPool2D
from parameterized import parameterized

np.random.seed(42)

class TestMaxPool2D(unittest.TestCase):
  @parameterized.expand([(2, 2), (3, 3), (2, 3), (3, 2)])
  def test_maxpool2d(self, kernel_w, kernel_h):
    img = np.random.normal(size=(8, 3, 32, 32)).astype(np.float32)
    kernel_size = (kernel_w, kernel_h)

    # Test forward
    my_img = Tensor(img)
    my_result = MaxPool2D.apply(my_img, kernel_size)
    torch_img = torch.tensor(img, requires_grad=True)
    torch_result = torch.nn.functional.max_pool2d(torch_img, kernel_size)
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy())

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_img.grad.numpy(), torch_img.grad.numpy())
