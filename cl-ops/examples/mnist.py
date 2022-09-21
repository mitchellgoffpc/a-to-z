import os
import gzip
import struct
import requests
import numpy as np

OUTPUT_DIR = '/tmp/mnist'
FILES = {
  'train-images-idx3': 9912422,
  'train-labels-idx1': 28881,
  't10k-images-idx3': 1648877,
  't10k-labels-idx1': 4542}
DTYPES = {
  0x08: np.uint8,
  0x09: np.int8,
  0x0B: np.int16,
  0x0C: np.int32,
  0x0D: np.float32,
  0x0E: np.float64}

def download_mnist():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  for fn, size in FILES.items():
    output_path = os.path.join(OUTPUT_DIR, f'{fn}-ubyte.gz')
    if not os.path.exists(output_path):
      print(f"Downloading {fn}-ubyte.gz ...")
      r = requests.get(f'http://yann.lecun.com/exdb/mnist/{fn}-ubyte.gz')
      with open(output_path, 'wb') as f:
        f.write(r.content)
    output_size = os.path.getsize(output_path)
    assert output_size == size, f"output size mismatch: {output_size} != {size}"

def load_file(fn):
  with gzip.open(fn) as f:
    _,_,dtype,ndims = f.read(4)
    dims = struct.unpack(f">{ndims}i", f.read(ndims*4))
    return np.frombuffer(f.read(), dtype=DTYPES[dtype]).reshape(dims)

def load_mnist():
  download_mnist()
  results = []
  for split in ('train', 't10k'):
    images = load_file(os.path.join(OUTPUT_DIR, f'{split}-images-idx3-ubyte.gz'))
    labels = load_file(os.path.join(OUTPUT_DIR, f'{split}-labels-idx1-ubyte.gz'))
    results.append((images, labels))
  return results

if __name__ == '__main__':
  train, test = load_mnist()
  print(train[0].shape, train[1].shape)
  print(test[0].shape, test[1].shape)
