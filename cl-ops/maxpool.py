import numpy as np
from helpers import Tensor, Function, clbuild, buffer_new, buffer_like

maxpool2d_forward = clbuild(f"""
  __kernel void maxpool2d_forward(__global const float *img_g, __global float *out_g, __global char *idxs_g,
                                  const int N, const int C, const int HIN, const int WIN, const int KH, const int KW) {{
    int nc = get_global_id(0), hout = get_global_id(1), wout = get_global_id(2);
    int n = nc / C, c = nc % C;
    int HOUT = HIN / KH, WOUT = WIN  / KW;
    int hin = hout * KH, win = wout * KW;
    float result = -FLT_MAX;
    char idx_h = 0, idx_w = 0;
    for (int h=0; h<KH; h++) {{
      for (int w=0; w<KW; w++) {{
        float x = img_g[n*C*WIN*HIN + c*WIN*HIN + (hin+h)*WIN + (win+w)];
        if (x > result) {{
          result = x;
          idx_h = h;
          idx_w = w;
        }}
      }}
    }}
    out_g[nc*HOUT*WOUT + hout*WOUT + wout] = result;
    idxs_g[nc*HOUT*WOUT + hout*WOUT + wout] = idx_h*KW + idx_w;
  }}""").maxpool2d_forward

maxpool2d_backward = clbuild(f"""
  __kernel void maxpool2d_backward(__global const float *d_out_g, __global const char *idxs_g, __global float *d_img_g,
                                   const int N, const int C, const int HIN, const int WIN, const int KH, const int KW) {{
    int nc = get_global_id(0), hout = get_global_id(1), wout = get_global_id(2);
    int n = nc / C, c = nc % C;
    int HOUT = HIN / KH, WOUT = WIN  / KW;
    int hin = hout * KH, win = wout * KW;
    float d_out = d_out_g[n*C*WOUT*HOUT + c*WOUT*HOUT + hout*WOUT + wout];
    char idx = idxs_g[n*C*WOUT*HOUT + c*WOUT*HOUT + hout*WOUT + wout];
    for (int h=0; h<KH; h++) {{
      for (int w=0; w<KW; w++) {{
        d_img_g[nc*HIN*WIN + (hin+h)*WIN + (win+w)] = d_out * (h*KW + w == idx);
      }}
    }}
  }}""").maxpool2d_backward

class MaxPool2D(Function):
  def forward(self, inputs, kernel_size):
    n,c,h,w = map(np.int32, inputs.shape)
    kh,kw = map(np.int32, kernel_size)
    out_shape = (n, c, h//kh, w//kw)
    output = buffer_new(out_shape)
    idxs = buffer_new(out_shape, dtype=np.dtype('uint8'))
    maxpool2d_forward(self.cl_queue, [n*c, h//kh, w//kw], None, inputs.data, output, idxs, n, c, h, w, kh, kw)
    self.save_for_backward(inputs, idxs, kernel_size)
    return Tensor(output, shape=out_shape)

  def backward(self, d_out):
    inputs, idxs, kernel_size = self.saved_tensors
    n,c,h,w = map(np.int32, inputs.shape)
    kh,kw = map(np.int32, kernel_size)
    d_inputs = buffer_like(inputs.data)
    maxpool2d_backward(self.cl_queue, [n*c, h//kh, w//kw], None, d_out.data, idxs, d_inputs, n, c, h, w, kh, kw)
    return [Tensor(d_inputs, shape=inputs.shape)]
