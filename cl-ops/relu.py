from helpers import Tensor, Function, clbuild, buffer_like

class Relu(Function):
  @staticmethod
  def forward(ctx, x):
    kernel = clbuild("""
    __kernel void relu_forward(__global const float *in, __global float *out) {
      int gid = get_global_id(0);
      out[gid] = max(in[gid], 0.0);
    }""")

    ctx.save_for_backward(x)
    output = buffer_like(x.data)
    kernel.relu_forward(ctx.cl_queue, [x.size], None, x.data, output)
    return Tensor(output, shape=x.shape)

  @staticmethod
  def backward(ctx, d_out):
    kernel = clbuild("""
    __kernel void relu_backward(__global const float *d_out, global const float *in, __global float *d_in) {
      int gid = get_global_id(0);
      d_in[gid] = in[gid] > 0 ? d_out[gid] : 0.0;
    }""")

    x, = ctx.saved_tensors
    d_in = buffer_like(x.data)
    kernel.relu_backward(ctx.cl_queue, [x.size], None, d_out.data, x.data, d_in)
    return [Tensor(d_in, shape=x.shape)]
