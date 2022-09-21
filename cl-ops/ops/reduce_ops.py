from helpers import Tensor, Function, clbuild, buffer_new, buffer_like

def reduce_op_forward(map_op, reduce_op):
  return clbuild(f"""
    __kernel void reduce_op_forward(__global const float *a_g, __global float *out_g) {{
      __local float a_local[256];
      int n = get_global_size(0);
      int gid = get_global_id(0);
      int tid = get_local_id(0);
      int group_id = get_group_id(0);
      int group_size = get_local_size(0);

      float a = a_g[gid];
      a_local[tid] = {map_op};
      barrier(CLK_LOCAL_MEM_FENCE);

      for (int stride = group_size / 2; stride > 0; stride >>= 1) {{
        float acc = a_local[tid];
        float a = a_local[tid + stride];
        a_local[tid] = {reduce_op};
        barrier(CLK_LOCAL_MEM_FENCE);
      }}
      if (tid == 0) {{
        out_g[group_id] = a_local[tid];
      }}
    }}""").reduce_op_forward

def reduce_op_backward(op):
  return clbuild(f"""
    __kernel void reduce_op_backward(__global const float *a_g,     __global const float *out_g,
                                     __global const float *d_out_g, __global float *d_a_g) {{
      int n = get_global_size(0);
      int gid = get_global_id(0);
      float a = a_g[gid];
      float out = out_g[0];
      float d_out = d_out_g[0];
      d_a_g[gid] = {op};
    }}""").reduce_op_backward

class ReduceOp(Function):
  def forward(self, a):
    output = buffer_new(shape=(1,))
    self.forward_kernel(self.cl_queue, [a.size], [16*16], a.data, output)
    result = Tensor(output, shape=(1,))
    self.save_for_backward(a, result)
    return result

  def backward(self, d_out):
    a, result = self.saved_tensors
    d_a = buffer_like(a.data)
    self.backward_kernel(self.cl_queue, [a.size], None, a.data, result.data, d_out.data, d_a)
    return [Tensor(d_a, shape=a.shape)]

class Sum(ReduceOp):
  forward_kernel = reduce_op_forward("a", "acc + a")
  backward_kernel = reduce_op_backward("d_out")

class Mean(ReduceOp):
  forward_kernel = reduce_op_forward("a / n", "acc + a")
  backward_kernel = reduce_op_backward("d_out / n")

class Min(ReduceOp):
  forward_kernel = reduce_op_forward("a", "min(acc, a)")
  backward_kernel = reduce_op_backward("a == out ? d_out : 0.0")

class Max(ReduceOp):
  forward_kernel = reduce_op_forward("a", "max(acc, a)")
  backward_kernel = reduce_op_backward("a == out ? d_out : 0.0")
