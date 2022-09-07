import glfw
import pyrr
import ctypes
import itertools
import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from contextlib import contextmanager
from pyrr import Matrix44, Vector3
from geometry import create_world_geometry, create_rotation_matrix, create_texture, load_texture_data

vertex_shader_source = """
# version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv;
out vec3 frag_normal;
out vec2 frag_uv;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    frag_normal = normal;
    frag_uv = uv;
}
"""

fragment_shader_source = """
# version 330 core
in vec3 frag_normal;
in vec2 frag_uv;
out vec4 out_color;
uniform sampler2D frag_texture;

void main(void) {
  out_color = texture(frag_texture, frag_uv);
}"""

MOVEMENT_KEYS = {
  glfw.KEY_W: np.array([1,0,0]),
  glfw.KEY_S: np.array([-1,0,0]),
  glfw.KEY_D: np.array([0,0,1]),
  glfw.KEY_A: np.array([0,0,-1]),
  glfw.KEY_SPACE: np.array([0,1,0]),
  glfw.KEY_LEFT_SHIFT: np.array([0,-1,0])}

ROTATION_KEYS = {
  glfw.KEY_Q: 1,
  glfw.KEY_E: -1}

KEYS = set()

@contextmanager
def create_window(key_callback=None):
  if not glfw.init():
    print("Cannot initialize GLFW")
    exit()

  glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
  glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
  glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
  glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
  glfw.window_hint(glfw.DEPTH_BITS, True)
  window = glfw.create_window(640, 480, "OpenGL window", None, None)
  if not window:
    print("GLFW window cannot be creted")
    glfw.terminate()
    exit()

  glfw.set_window_pos(window, 100, 100)
  glfw.make_context_current(window)
  if key_callback:
    glfw.set_key_callback(window, key_callback)

  try: yield window
  finally: glfw.terminate()

def create_buffer(data):
  buffer_id = GL.glGenBuffers(1)
  GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
  GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
  return buffer_id

def bind_buffer(program, attribute, buffer_id, stride):
  attribute_location = GL.glGetAttribLocation(program, attribute)
  GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
  GL.glVertexAttribPointer(attribute_location, stride, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
  GL.glEnableVertexAttribArray(attribute_location)

def get_camera_matrix(position, rotation):
  view_matrix = np.transpose(pyrr.matrix44.create_look_at(
    position,
    position + np.array([1, -1, 0], dtype=np.float32) @ create_rotation_matrix(rotation),
    np.array([0.0, 1.0, 0.0], dtype=np.float32)))
  proj_matrix = np.transpose(pyrr.matrix44.create_perspective_projection(45.0, 640/480, 0.1, 200.0))
  return proj_matrix @ view_matrix

def get_mvp_matrix(translation, rotation, scale):
  trans_matrix = np.transpose(pyrr.matrix44.create_from_translation(translation))
  rot_matrix = np.transpose(pyrr.matrix44.create_from_y_rotation(rotation))
  scale_matrix = np.transpose(pyrr.matrix44.create_from_scale(scale))
  return trans_matrix @ rot_matrix @ scale_matrix

def get_movement():
  position = np.zeros(3, dtype=np.float32)
  for key in KEYS:
    if key in MOVEMENT_KEYS:
      position += MOVEMENT_KEYS[key] / 10
  return position

def get_rotation():
  theta = 0
  for key in KEYS:
    if key in ROTATION_KEYS:
      theta += ROTATION_KEYS[key] / 60
  return theta


def main(window):
  # Initialize the VAO
  vao = GL.glGenVertexArrays(1)
  GL.glBindVertexArray(vao)

  # Initialize the shaders
  vertex_shader = shaders.compileShader(vertex_shader_source, GL.GL_VERTEX_SHADER)
  fragment_shader = shaders.compileShader(fragment_shader_source, GL.GL_FRAGMENT_SHADER)
  program = shaders.compileProgram(vertex_shader, fragment_shader)

  # Load the position buffer and attributes
  vertex_data, uv_data = create_world_geometry('01')
  buffer_data = {'position': vertex_data, 'uv': uv_data, 'normal': vertex_data}
  buffer_strides = {'position': 3, 'uv': 2, 'normal': 3}
  buffers = {key: create_buffer(data) for key, data in buffer_data.items()}
  for key, buffer in buffers.items():
    bind_buffer(program, key, buffer, buffer_strides[key])

  # Load the texture
  texture_id = create_texture(load_texture_data())
  texture_attr = GL.glGetUniformLocation(program, "frag_texture")
  GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

  # A bit of opengl setup
  GL.glClearColor(0.95, 1.0, 0.95, 0)
  GL.glEnable(GL.GL_DEPTH_TEST)
  GL.glUseProgram(program)
  mvp = GL.glGetUniformLocation(program, "mvp")

  # Initialize the position and rotation
  position = np.array([-3, 8, 4], dtype=np.float32)
  rotation = 0

  while not glfw.window_should_close(window):
    rotation += get_rotation()
    position += get_movement() @ create_rotation_matrix(rotation)
    camera_matrix = get_camera_matrix(position, rotation)
    GL.glUniformMatrix4fv(mvp, 1, GL.GL_FALSE, camera_matrix.T)

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(vertex_data) // 3)
    glfw.poll_events()
    glfw.swap_buffers(window)

  glfw.terminate()


if __name__ == '__main__':
  def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS:
      KEYS.add(key)
    elif action == glfw.RELEASE:
      KEYS.remove(key)

  with create_window(key_callback) as window:
    main(window)
