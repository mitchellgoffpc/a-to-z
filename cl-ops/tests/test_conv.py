import torch
import unittest
import numpy as np
from helpers import Tensor
from conv import Conv2D

np.random.seed(42)

class TestConv2D(unittest.TestCase):
  def test_conv2d(self):
    img = np.random.normal(size=(8, 3, 32, 32)).astype(np.float32)
    weight = np.random.normal(size=(16, 3, 1, 1)).astype(np.float32)

    # Test forward
    my_img, my_weight = Tensor(img), Tensor(weight)
    my_result = Conv2D.apply(my_img, my_weight)
    torch_img = torch.tensor(img, requires_grad=True)
    torch_weight = torch.tensor(weight, requires_grad=True)
    torch_result = torch.nn.functional.conv2d(torch_img, torch_weight)
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy())

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_img.grad.numpy(), torch_img.grad.numpy())
    # np.testing.assert_allclose(my_weight.grad.numpy(), torch_weight.grad.numpy())
