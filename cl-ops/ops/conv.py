import numpy as np
from ops.pad import Pad2D
from helpers import Tensor, Function, clbuild, buffer_new, buffer_like

conv2d_forward = clbuild(f"""
  __kernel void conv2d_forward(__global const float *img_g, __global const float *weight_g, __global float *out_g,
                               const int N, const int CIN, const int COUT, const int HIN, const int WIN, const int KH, const int KW) {{
    int ncout = get_global_id(0), hout = get_global_id(1), wout = get_global_id(2);
    int n = ncout / COUT, cout = ncout % COUT;
    int HOUT = HIN - KH + 1, WOUT = WIN - KW + 1;

    float result = 0;
    for (int cin = 0; cin < CIN; cin++) {{
      for (int kh = 0; kh < KH; kh++) {{
        for (int kw = 0; kw < KW; kw++) {{
          int hin = hout + kh, win = wout + kw;
          result += img_g[n*CIN*WIN*HIN + cin*WIN*HIN + hin*WIN + win] * weight_g[cout*CIN*KH*KW + cin*KH*KW + kh*KW + kw];
        }}
      }}
    }}
    out_g[ncout*HOUT*WOUT + hout*WOUT + wout] = result;
  }}""").conv2d_forward

conv2d_backward = clbuild(f"""
  __kernel void conv2d_backward(__global const float *img_g, __global const float *weight_g, __global const float *out_g,
                                __global const float *d_out_g, __global float *d_img_g, __global float *d_weight_g,
                               const int N, const int CIN, const int COUT, const int HIN, const int WIN, const int KH, const int KW) {{
    int ncin = get_global_id(0), hin = get_global_id(1), win = get_global_id(2);
    int n = ncin / CIN, cin = ncin % CIN;
    int HOUT = HIN - KH + 1, WOUT = WIN - KW + 1;

    float d_img = 0, d_weight = 0;
    for (int cout = 0; cout < COUT; cout++) {{
      for (int kh = 0; kh < KH; kh++) {{
        for (int kw = 0; kw < KW; kw++) {{
          int hout = hin, wout = win; // + kh - KH, wout = win + kw - KW;
          float d_out = d_out_g[n*COUT*HOUT*WOUT + cout*HOUT*WOUT + hout*WOUT + wout];
          d_img += d_out * weight_g[cout*CIN*KH*KW + cin*KH*KW + (KH-kh-1)*KW + (KW-kw-1)];
          d_weight += d_out * img_g[n*CIN*WIN*HIN + cin*WIN*HIN + hin*WIN + win];
        }}
      }}
    }}
    d_img_g[ncin*HIN*WIN + hin*WIN + win] = d_img;
    d_weight_g[ncin*HIN*WIN + hin*WIN + win] = d_weight;
  }}""").conv2d_backward

class Conv2D(Function):
  def forward(self, inputs, weights):
    n,cin,h,w = map(np.int32, inputs.shape)
    cout,cin,kh,kw = map(np.int32, weights.shape)
    out_shape = (n,cout,h-kh+1,w-kw+1)
    output = buffer_new(out_shape)
    conv2d_forward(self.cl_queue, [n*cout, h-kh+1, w-kw+1], None, inputs.data, weights.data, output, n, cin, cout, h, w, kh, kw)
    result = Tensor(output, shape=out_shape)
    self.save_for_backward(inputs, weights, result)
    return result

  def backward(self, d_out):
    inputs, weights, result = self.saved_tensors
    n,cin,h,w = map(np.int32, inputs.shape)
    cout,cin,kh,kw = map(np.int32, weights.shape)
    # d_out = Pad2D.apply(d_out, (kh-1,kh-1,kw-1,kw-1))
    d_inputs = buffer_like(inputs.data)
    d_weights = buffer_like(weights.data)
    conv2d_backward(self.cl_queue, [n*cin, h, w], None,
                    inputs.data, weights.data, result.data, d_out.data, d_inputs, d_weights,
                    n, cin, cout, h, w, kh, kw)
    return [Tensor(d_inputs, shape=inputs.shape), Tensor(d_weights, shape=weights.shape)]
