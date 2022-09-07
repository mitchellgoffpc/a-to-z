import cv2
import numpy as np
from OpenGL import GL
from assets import Tiles, load_image, load_map, load_tiles

TEXTURE_W = 9
TEXTURE_H = 3

# Create OpenGL data

tile_uvs = np.array([
  1, 1,
  0, 0,
  1, 0,
  1, 1,
  0, 1,
  0, 0,
], dtype=np.float32)

inside_corner_uvs = np.array([
  1, 1,
  0, 0,
  1, 0,
  1, 1,
  1, 0,
  0, 0,
], dtype=np.float32)

outside_corner_uvs = np.array([
  1, 1,
  0, 0,
  0, 1,
  1, 1,
  0, 1,
  0, 0,
], dtype=np.float32)

def create_tile(x, y, z):
  return np.array([
    x,   y, z,
    x+1, y, z+1,
    x,   y, z+1,
    x,   y, z,
    x+1, y, z,
    x+1, y, z+1
  ], dtype=np.float32)

def create_wall(x, z, theta):
  wall = np.array([
     0.5, 0,  0.5,
    -0.5, 1, -0.5,
     0.5, 1, -0.5,
     0.5, 0,  0.5,
    -0.5, 0,  0.5,
    -0.5, 1, -0.5
  ], dtype=np.float32)
  return rotate_and_translate(wall, x+0.5, z+0.5, theta)

def create_inside_corner(x, z, theta):
  wall = np.array([
     0.5, 0,  0.5,
    -0.5, 1, -0.5,
     0.5, 1, -0.5,
     0.5, 0,  0.5,
    -0.5, 1,  0.5,
    -0.5, 1, -0.5
  ], dtype=np.float32)
  return rotate_and_translate(wall, x+0.5, z+0.5, theta)

def create_outside_corner(x, z, theta):
  wall = np.array([
    -0.5, 0,  0.5,
     0.5, 1, -0.5,
     0.5, 0,  0.5,
    -0.5, 0,  0.5,
    -0.5, 0, -0.5,
     0.5, 1, -0.5
  ], dtype=np.float32)
  return rotate_and_translate(wall, x+0.5, z+0.5, theta)

def load_texture_data():
  texture_data = np.zeros((128*TEXTURE_H, 128*TEXTURE_W, 3), dtype=np.uint8)
  for i, world_type in enumerate(['Rock', 'Ice', 'Lava']):
    for j, idx in enumerate((*range(6), 45, 46)):
      texture_data[i*128:i*128+128, j*128:j*128+128] = load_image(f'World/WorldTextures/{world_type}Split/{world_type.upper()}{idx:02}.bmp')
    roof_img = load_image(f'World/WorldTextures/{world_type.upper()}ROOF.bmp')
    texture_data[i*128:i*128+128:, -128:] = cv2.resize(roof_img, (128, 128))
  return texture_data

def create_texture(img):
  texture_id = GL.glGenTextures(1)
  GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
  GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
  GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, img.shape[1], img.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img.tobytes())
  GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
  return texture_id

def create_rotation_matrix(theta):
  return np.array([
    [np.cos(theta), 0, -np.sin(theta)],
    [0,             1,  0],
    [np.sin(theta), 0, np.cos(theta)]
  ], dtype=np.float32)

def rotate_and_translate(vertices, x, z, theta):
  vertices = vertices.reshape(-1, 3) @ create_rotation_matrix(theta)
  vertices = vertices + np.array([x, 0, z], dtype=np.float32)
  return vertices.flatten()

def get_uv_coords(uvs, y, x):
  scale = np.array([[1/TEXTURE_W, 1/TEXTURE_H]], dtype=np.float32)
  return uvs.reshape(6, 2) * scale + np.array([x, y], dtype=np.float32) * scale


# Map generation

def is_floor_tile(x):
  return x in (Tiles.FLOOR, Tiles.WATER, Tiles.LAVA)

def in_bounds(map, y, x):
  return 0 <= y < map.shape[0] and 0 <= x < map.shape[1]

def create_world_geometry(level, world_type=0):
  map = load_tiles(level)
  height = load_map(level, map='High')
  vertices, uvs = [], []
  for y in range(map.shape[0]):
    for x in range(map.shape[1]):
      if is_floor_tile(map[y,x]):
        vertices.append(create_tile(x, 0, y))
        uvs.append(get_uv_coords(tile_uvs, world_type, {Tiles.WATER: 6, Tiles.LAVA: 7}.get(map[y,x], 0)))
      else:
        tile_idx = 6 - map[y,x]
        adjacent_pos = [(y+1,x), (y,x+1), (y-1,x), (y,x-1)]
        adjacent_floors = [in_bounds(map, dy, dx) and is_floor_tile(map[dy,dx]) for dy,dx in adjacent_pos]
        diagonal_pos = [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]
        diagonal_floors = [in_bounds(map, dy, dx) and is_floor_tile(map[dy,dx]) for dy,dx in diagonal_pos]
        if not any(adjacent_floors) and not any(diagonal_floors):  # Ceiling
          vertices.append(create_tile(x, 1, y))
          uvs.append(get_uv_coords(tile_uvs, world_type, 8))
        elif not any(adjacent_floors) and sum(diagonal_floors) == 1:  # Inside corner
          vertices.append(create_inside_corner(x, y, np.pi - np.pi/2 * np.argmax(diagonal_floors)))
          uvs.append(get_uv_coords(inside_corner_uvs, world_type, tile_idx))
        elif sum(adjacent_floors) == 1:  # Straight wall
          vertices.append(create_wall(x, y, np.pi/2 * np.argmax(adjacent_floors)))
          uvs.append(get_uv_coords(tile_uvs, world_type, tile_idx))
        elif sum(adjacent_floors) == 2:
          rotation = 3 if adjacent_floors[0] and adjacent_floors[3] else np.argmax(adjacent_floors)
          vertices.append(create_outside_corner(x, y, np.pi/2 + np.pi/2 * rotation))
          uvs.append(get_uv_coords(outside_corner_uvs, world_type, tile_idx))
        else:
          vertices.append(create_tile(x, 0, y))
          uvs.append(get_uv_coords(tile_uvs, world_type, 0))

  return np.concatenate(vertices), np.concatenate(uvs)
