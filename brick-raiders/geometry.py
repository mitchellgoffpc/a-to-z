import numpy as np

class Tiles:
  FLOOR      = 0
  WATER      = 1
  LAVA       = 2
  DIRT       = 3
  LOOSE_ROCK = 4
  HARD_ROCK  = 5
  SOLID_ROCK = 6

color_data = np.array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
  1, 0, 0,
  0, 1, 0,
  0, 0, 1
], dtype=np.float32)


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
     0.5, 0,  0.5,
    -0.5, 0,  0.5,
     0.5, 1, -0.5,
    -0.5, 0,  0.5,
    -0.5, 0, -0.5,
     0.5, 1, -0.5
  ], dtype=np.float32)
  print(x, z, theta)
  return rotate_and_translate(wall, x+0.5, z+0.5, theta)

def is_floor_tile(x):
  return x in (Tiles.FLOOR, Tiles.WATER, Tiles.LAVA)

def in_bounds(map, y, x):
  return 0 <= y < map.shape[0] and 0 <= x < map.shape[1]

def create_world_geometry(map):
  vertices, colors = [], []
  for y in range(map.shape[0]):
    for x in range(map.shape[1]):
      if is_floor_tile(map[y,x]):
        vertices.append(create_tile(x, 0, y))
        colors.append(color_data)
      else:
        adjacent_pos = [(y+1,x), (y,x+1), (y-1,x), (y,x-1)]
        adjacent_floors = [in_bounds(map, dy, dx) and is_floor_tile(map[dy,dx]) for dy,dx in adjacent_pos]
        diagonal_pos = [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]
        diagonal_floors = [in_bounds(map, dy, dx) and is_floor_tile(map[dy,dx]) for dy,dx in diagonal_pos]
        if not any(adjacent_floors) and not any(diagonal_floors):  # Ceiling
          vertices.append(create_tile(x, 1, y))
          colors.append(color_data * .2)
        elif not any(adjacent_floors) and sum(diagonal_floors) == 1:  # Inside corner
          vertices.append(create_inside_corner(x, y, np.pi - np.pi/2 * np.argmax(diagonal_floors)))
          colors.append(color_data)
        elif sum(adjacent_floors) == 1:  # Straight wall
          vertices.append(create_wall(x, y, np.pi/2 * np.argmax(adjacent_floors)))
          colors.append(color_data)
        elif sum(adjacent_floors) == 2:
          rotation = 3 if adjacent_floors[0] and adjacent_floors[3] else np.argmax(adjacent_floors)
          vertices.append(create_outside_corner(x, y, np.pi/2 + np.pi/2 * rotation))
          colors.append(color_data)

  return np.concatenate(vertices), np.concatenate(colors)
