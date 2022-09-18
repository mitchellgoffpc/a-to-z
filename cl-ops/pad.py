import numpy as np
from helpers import Tensor, Function, clbuild, buffer_new, buffer_like

pad2d_forward = clbuild(f"""
  __kernel void pad2d_forward(__global const float *img_g, __global float *out_g,
                              const int N, const int C, const int HIN, const int WIN,
                              const int H1, const int H2, const int W1, const int W2) {{
    int nc = get_global_id(0), hin = get_global_id(1), win = get_global_id(2);
    int n = nc / C, c = nc % C;
    int HOUT = HIN + H1 + H2, WOUT = WIN + W1 + W2;
    int hout = hin + H1, wout = win + W1;
    out_g[nc*HOUT*WOUT + hout*WOUT + wout] = img_g[n*C*WIN*HIN + c*WIN*HIN + hin*WIN + win];
  }}""").pad2d_forward

crop2d_forward = clbuild(f"""
  __kernel void crop2d_forward(__global const float *img_g, __global float *out_g,
                               const int N, const int C, const int HIN, const int WIN,
                               const int H1, const int H2, const int W1, const int W2) {{
    int nc = get_global_id(0), hout = get_global_id(1), wout = get_global_id(2);
    int n = nc / C, c = nc % C;
    int HOUT = HIN - H1 - H2, WOUT = WIN - W1 - W2;
    int hin = hout + H1, win = wout + W1;
    out_g[nc*HOUT*WOUT + hout*WOUT + wout] = img_g[n*C*WIN*HIN + c*WIN*HIN + hin*WIN + win];
  }}""").crop2d_forward

class Pad2D(Function):
  def forward(self, inputs, padding):
    n,c,h,w = map(np.int32, inputs.shape)
    h1,h2,w1,w2 = map(np.int32, padding)
    out_shape = (n, c, h+h1+h2, w+w1+w2)
    output = buffer_new(out_shape)
    pad2d_forward(self.cl_queue, [n*c, h, w], None, inputs.data, output, n, c, h, w, h1, h2, w1, w2)
    self.save_for_backward(inputs, padding)
    return Tensor(output, shape=out_shape)

  def backward(self, d_out):
    inputs, padding = self.saved_tensors
    n,c,h,w = map(np.int32, inputs.shape)
    h1,h2,w1,w2 = map(np.int32, padding)
    d_inputs = buffer_like(inputs.data)
    crop2d_forward(self.cl_queue, [n*c, h, w], None, d_out.data, d_inputs, n, c, h+h1+h2, w+w1+w2, h1, h2, w1, w2)
    return [Tensor(d_inputs, shape=inputs.shape)]

class Crop2D(Function):
  def forward(self, inputs, padding):
    n,c,h,w = map(np.int32, inputs.shape)
    h1,h2,w1,w2 = map(np.int32, padding)
    out_shape = (n, c, h-h1-h2, w-w1-w2)
    output = buffer_new(out_shape)
    crop2d_forward(self.cl_queue, [n*c, h-h1-h2, w-w1-w2], None, inputs.data, output, n, c, h, w, h1, h2, w1, w2)
    self.save_for_backward(inputs, padding)
    return Tensor(output, shape=out_shape)

  def backward(self, d_out):
    inputs, padding = self.saved_tensors
    n,c,h,w = map(np.int32, inputs.shape)
    h1,h2,w1,w2 = map(np.int32, padding)
    d_inputs = buffer_like(inputs.data)
    pad2d_forward(self.cl_queue, [n*c, h-h1-h2, w-w1-w2], None, d_out.data, d_inputs, n, c, h-h1-h2, w-w1-w2, h1, h2, w1, w2)
    return [Tensor(d_inputs, shape=inputs.shape)]
