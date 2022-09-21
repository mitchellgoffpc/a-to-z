import torch
import unittest
import numpy as np
from helpers import Tensor
from ops.pad import Pad2D, Crop2D
from parameterized import parameterized

np.random.seed(42)

class TestPad2D(unittest.TestCase):
  @parameterized.expand([(0,), (1,), (2,)])
  def test_pad2d(self, p):
    img = np.random.normal(size=(8, 3, 32, 32)).astype(np.float32)

    # Test forward
    my_img = Tensor(img)
    my_result = Pad2D.apply(my_img, (p,p,p,p))
    torch_img = torch.tensor(img, requires_grad=True)
    torch_result = torch.nn.functional.pad(torch_img, (p,p,p,p))
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy())

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_img.grad.numpy(), torch_img.grad.numpy())

class TestCrop2D(unittest.TestCase):
  @parameterized.expand([(0,), (1,), (2,)])
  def test_crop2d(self, p):
    img = np.random.normal(size=(8, 3, 32, 32)).astype(np.float32)

    # Test forward
    my_img = Tensor(img)
    my_result = Crop2D.apply(my_img, (p,p,p,p))
    torch_img = torch.tensor(img, requires_grad=True)
    torch_result = torch.nn.functional.pad(torch_img, (-p,-p,-p,-p))
    np.testing.assert_allclose(my_result.numpy(), torch_result.detach().numpy())

    # Test backward
    my_result.backward()
    torch_result.sum().backward()
    np.testing.assert_allclose(my_img.grad.numpy(), torch_img.grad.numpy())
