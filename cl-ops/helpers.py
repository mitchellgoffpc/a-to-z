import math, functools
import numpy as np
import pyopencl as cl

cl_ctx = cl.Context(dev_type=cl.device_type.ALL)
cl_queue = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
flags = cl.mem_flags

class Tensor:
  def __init__(self, data, shape=None):
    self.ctx = None
    self.grad = None
    if isinstance(data, np.ndarray):
      self.shape = shape or data.shape
      self.data = cl.Buffer(cl_ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=data)
    elif isinstance(data, cl.Buffer):
      self.data = data
      self.shape = shape
    else:
      raise NotImplementedError(f"Tensor constructor not implemented for <{data.__class__}>")

  def numpy(self):
    data_np = np.zeros(self.shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data_np, self.data)
    return data_np

  def toposort(self):
    def _toposort(node, visited=set(), result=[]):
      visited.add(node)
      if self.ctx:
        for x in self.ctx.inputs:
          if x not in visited:
            _toposort(x, visited, result)
        result.append(node)
      return result
    return list(reversed(_toposort(self)))

  def backward(self):
    self.grad = Tensor(np.ones(self.shape, np.float32))
    for node in self.toposort():
      if node.ctx:
        for x, grad in zip(node.ctx.inputs, node.ctx.backward(self.grad)):
          x.grad = grad if x.grad is None else Add.apply(x.grad, grad)
      del node.ctx

  @property
  def size(self):
    return math.prod(self.shape)

class Function:
  def __init__(self, *inputs):
    self.inputs = [x for x in inputs if isinstance(x, Tensor)]
    self.cl_ctx = cl_ctx
    self.cl_queue = cl_queue
    self.saved_tensors = []

  @classmethod
  def apply(cls, *args):
    ctx = cls(*args)
    result = ctx.forward(*args)
    result.ctx = ctx
    return result

  def save_for_backward(self, *args):
    self.saved_tensors.extend(args)

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f"forward not implemented for <{self.__class__}>")
  def backward(self, *args, **kwargs):
    raise NotImplementedError(f"backward not implemented for <{self.__class__}>")



# Helper functions

@functools.lru_cache()
def clbuild(source):
  return cl.Program(cl_ctx, source).build()

def buffer_new(shape, dtype=np.dtype('float32')):
  return cl.Buffer(cl_ctx, flags.READ_ONLY, size=np.prod(shape)*dtype.itemsize)

def buffer_like(buffer):
  return cl.Buffer(cl_ctx, flags.READ_ONLY, size=buffer.size)
