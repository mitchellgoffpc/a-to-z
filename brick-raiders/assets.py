import os
import struct
import numpy as np
from PIL import Image

BASE_DIR = '/Users/mitchell/Downloads/Rock Raiders/Data/Assets'

class Tiles:
  FLOOR         = 0
  SOLID_ROCK    = 1
  HARD_ROCK     = 2
  LOOSE_ROCK    = 3
  DIRT          = 4
  DIRT2         = 5
  LAVA          = 6
  UNUSED        = 7
  ORE_SEAM      = 8
  WATER         = 9
  ENERGY_SEAM   = 10
  RECHARGE_SEAM = 11

def load_map(level='01', map='Surf'):
  with open(os.path.join(BASE_DIR, 'Levels/GameLevels', f'Level{level}', f'{map}_{level}.map'), 'rb') as f:
    header, length, x, y = struct.unpack('4i', f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)
    assert np.all(data[1::2] == 0)
    return data[::2].reshape(y, x)

def load_tiles(level='01'):
  surf = load_map(level, map='Surf')
  dugg = load_map(level, map='Dugg')
  exposed = dugg & 0x1
  floor_tile = np.where((surf == Tiles.WATER) | (surf == Tiles.LAVA), surf, 0)
  return np.where(exposed, floor_tile, surf)

def load_image(filename):
  img = Image.open(os.path.join(BASE_DIR, filename))
  return np.array(img.convert("RGB"))
