from helpers import Tensor, Function, clbuild, buffer_like

def unary_op_forward(op):
  return clbuild(f"""
    __kernel void unary_op_forward(__global const float *a_g, __global float *out) {{
      int gid = get_global_id(0);
      float a = a_g[gid];
      out[gid] = {op};
    }}""").unary_op_forward

def unary_op_backward(op):
  return clbuild(f"""
    __kernel void unary_op_backward(__global const float *a_g,     __global const float *out_g,
                                     __global const float *d_out_g, __global float *d_a) {{
      int gid = get_global_id(0);
      float a = a_g[gid];
      float out = out_g[gid];
      float d_out = d_out_g[gid];
      d_a[gid] = {op};
    }}""").unary_op_backward

class UnaryOp(Function):
  def forward(self, a):
    output = buffer_like(a.data)
    self.forward_kernel(self.cl_queue, [a.size], None, a.data, output)
    result = Tensor(output, shape=a.shape)
    self.save_for_backward(a, result)
    return result

  def backward(self, d_out):
    a, result = self.saved_tensors
    d_a = buffer_like(a.data)
    self.backward_kernel(self.cl_queue, [a.size], None, a.data, result.data, d_out.data, d_a)
    return [Tensor(d_a, shape=a.shape)]

class Negate(UnaryOp):
  forward_kernel = unary_op_forward('-a')
  backward_kernel = unary_op_backward('-d_out')

class Relu(UnaryOp):
  forward_kernel = unary_op_forward('max(a, 0.0)')
  backward_kernel = unary_op_backward('a > 0 ? d_out : 0.0')

# Add operations to Tensor
for key, op in {'__neg__': Negate}.items():
  setattr(Tensor, key, op.apply)
