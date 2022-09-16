from helpers import Tensor, Function, clbuild, buffer_like

def binary_op_forward(op):
  return clbuild(f"""
    __kernel void binary_op_forward(__global const float *a_g, __global const float *b_g, __global float *out) {{
      int gid = get_global_id(0);
      float a = a_g[gid];
      float b = b_g[gid];
      out[gid] = {op};
    }}""").binary_op_forward

def binary_op_backward(op_a, op_b):
  return clbuild(f"""
    __kernel void binary_op_backward(__global const float *a_g,     __global const float *b_g, __global const float *out_g,
                                     __global const float *d_out_g, __global float *d_a,       __global float *d_b) {{
      int gid = get_global_id(0);
      float a = a_g[gid];
      float b = b_g[gid];
      float out = out_g[gid];
      float d_out = d_out_g[gid];
      d_a[gid] = {op_a};
      d_b[gid] = {op_b};
    }}""").binary_op_backward

class BinaryOp(Function):
  def forward(self, a, b):
    output = buffer_like(a.data)
    self.forward_kernel(self.cl_queue, [a.size], None, a.data, b.data, output)
    result = Tensor(output, shape=a.shape)
    self.save_for_backward(a, b, result)
    return result

  def backward(self, d_out):
    a, b, result = self.saved_tensors
    d_a = buffer_like(a.data)
    d_b = buffer_like(b.data)
    self.backward_kernel(self.cl_queue, [a.size], None, a.data, b.data, result.data, d_out.data, d_a, d_b)
    return [Tensor(d_a, shape=a.shape), Tensor(d_b, shape=b.shape)]

class Add(BinaryOp):
  forward_kernel = binary_op_forward("a + b")
  backward_kernel = binary_op_backward("d_out", "d_out")

class Sub(BinaryOp):
  forward_kernel = binary_op_forward("a - b")
  backward_kernel = binary_op_backward("d_out", "-d_out")

class Mul(BinaryOp):
  forward_kernel = binary_op_forward("a * b")
  backward_kernel = binary_op_backward("d_out * b", "d_out * a")

class Div(BinaryOp):
  forward_kernel = binary_op_forward("a / b")
  backward_kernel = binary_op_backward("d_out / b", "-d_out * a / (b * b)")

class Pow(BinaryOp):
  forward_kernel = binary_op_forward("pow(a, b)")
  backward_kernel = binary_op_backward("d_out * b * pow(a, b-1)", "d_out * out * log(a)")

# Add operations to Tensor
for key, op in {'__add__': Add, '__sub__': Sub, '__mul__': Mul, '__div__': Div, '__pow__': Pow}.items():
  setattr(Tensor, key, op.apply)
