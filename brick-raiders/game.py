import glfw
import pyrr
import ctypes
import itertools
import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from contextlib import contextmanager
from pyrr import Matrix44, Vector3
from geometry import create_world_geometry, create_rotation_matrix

vertex_shader_source = """
# version 330 core
in vec3 position;
in vec3 color;
in vec3 normal;
out vec3 frag_color;
out vec3 frag_normal;
uniform mat4 mvp;

void main() {
    frag_color = color;
    frag_normal = normal;
    gl_Position = mvp * vec4(position, 1.0);
}
"""

fragment_shader_source = """
# version 330 core
in vec3 frag_color;
in vec3 frag_normal;
out vec4 out_color;

void main(void) {
    out_color = vec4(frag_color, 1.0);
}"""

# fragment_shader_source = """
# # version 330 core
# in vec3 frag_color;
# in vec3 frag_normal;
# out vec4 out_color;
#
# void main() {
#     vec3 light_direction = normalize(vec3(0.3, 1.0, 0.3));
#     vec3 light_color = vec3(1.0, 1.0, 1.0);
#     vec3 ambient = vec3(0.5, 0.5, 0.5);
#
#     vec3 diffuse = max(dot(frag_normal, light_direction), 0.0) * light_color;
#     out_color = vec4((ambient + diffuse) * frag_color, 1);
# }
# """

map = np.array([
  [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
  [6, 6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 6],
  [6, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6],
  [6, 6, 6, 2, 2, 0, 0, 0, 0, 0, 0, 6],
  [6, 6, 6, 2, 2, 2, 0, 0, 0, 6, 6, 6],
  [6, 6, 6, 2, 2, 2, 2, 2, 0, 6, 6, 6],
  [6, 6, 6, 2, 2, 2, 2, 2, 2, 6, 6, 6],
  [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
], dtype=int)


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

def bind_buffer(program, attribute, buffer_id):
  attribute_location = GL.glGetAttribLocation(program, attribute)
  GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
  GL.glVertexAttribPointer(attribute_location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
  GL.glEnableVertexAttribArray(attribute_location)

def get_camera_matrix(position, rotation):
  view_matrix = np.transpose(pyrr.matrix44.create_look_at(
    position,
    # position + np.array([1, -1.5, 1], dtype=np.float32),
    np.array([6, 0, 4], dtype=np.float32),
    np.array([0.0, 1.0, 0.0], dtype=np.float32)))
  proj_matrix = np.transpose(pyrr.matrix44.create_perspective_projection(45.0, 640/480, 0.1, 200.0))
  return proj_matrix @ view_matrix

def get_mvp_matrix(translation, rotation, scale):
  trans_matrix = np.transpose(pyrr.matrix44.create_from_translation(translation))
  rot_matrix = np.transpose(pyrr.matrix44.create_from_y_rotation(rotation))
  scale_matrix = np.transpose(pyrr.matrix44.create_from_scale(scale))
  return trans_matrix @ rot_matrix @ scale_matrix


def main(window):
  # Initialize the VAO
  vao = GL.glGenVertexArrays(1)
  GL.glBindVertexArray(vao)

  # Initialize the shaders
  vertex_shader = shaders.compileShader(vertex_shader_source, GL.GL_VERTEX_SHADER)
  fragment_shader = shaders.compileShader(fragment_shader_source, GL.GL_FRAGMENT_SHADER)
  program = shaders.compileProgram(vertex_shader, fragment_shader)

  # Load the position buffer and attributes
  vertex_data, color_data = create_world_geometry(map)
  buffer_data = {'position': vertex_data, 'color': color_data, 'normal': vertex_data}
  buffers = {key: create_buffer(data) for key, data in buffer_data.items()}
  for key, buffer in buffers.items():
    bind_buffer(program, key, buffer)

  # Get ready for the render loop
  GL.glClearColor(0.95, 1.0, 0.95, 0)
  GL.glEnable(GL.GL_DEPTH_TEST)
  GL.glUseProgram(program)
  mvp = GL.glGetUniformLocation(program, "mvp")

  for fidx in itertools.count():
    if glfw.window_should_close(window): break
    theta = fidx / 360 * np.pi
    position = np.array([-6, 8, -6], dtype=np.float32) @ create_rotation_matrix(theta)
    position = np.array([6, 0, 4], dtype=np.float32) + position
    camera_matrix = get_camera_matrix(position, Vector3([0, 0, 0]))
    GL.glUniformMatrix4fv(mvp, 1, GL.GL_FALSE, camera_matrix.T)

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(vertex_data) // 3)
    glfw.poll_events()
    glfw.swap_buffers(window)

  glfw.terminate()


if __name__ == '__main__':
    with create_window() as window:
        main(window)
