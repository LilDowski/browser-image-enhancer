import type { EnhanceParams } from '../api/types';

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  // Переворот по Y делаем здесь, в текстурных координатах: это детерминированно и
  // не зависит от особенностей UNPACK_FLIP_Y/convertToBlob в разных браузерах.
  // Без него результат выгружается перевёрнутым по вертикали.
  v_uv = vec2((a_pos.x + 1.0) * 0.5, 1.0 - (a_pos.y + 1.0) * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;
void main() {
  vec4 t = texture2D(u_tex, v_uv);
  vec3 c = t.rgb;
  // 1) Яркость
  c += u_brightness;
  // 2) Контраст (относительно средней точки 0.5)
  c = (c - 0.5) * u_contrast + 0.5;
  // 3) Насыщенность (интерполяция к яркостной компоненте)
  float Y = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(Y), c, u_saturation);
  // 4) Ограничение диапазона
  c = clamp(c, 0.0, 1.0);
  gl_FragColor = vec4(c, t.a);
}`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('Ошибка компиляции шейдера: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

/**
 * Применяет параметры коррекции к изображению в полном разрешении на GPU.
 * Возвращает OffscreenCanvas с результатом, либо null, если WebGL недоступен.
 */
export function applyWebGL(source: ImageBitmap, p: EnhanceParams): OffscreenCanvas | null {
  const canvas = new OffscreenCanvas(source.width, source.height);
  const gl = canvas.getContext('webgl', {
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  }) as WebGLRenderingContext | null;
  if (!gl) return null;

  try {
    const program = gl.createProgram()!;
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error('Ошибка линковки программы: ' + gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);

    // Полноэкранный четырёхугольник (triangle strip)
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const aPos = gl.getAttribLocation(program, 'a_pos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Текстура из исходного изображения. UNPACK_FLIP_Y не используем — переворот
    // по Y делается в вершинном шейдере (см. VERT).
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    gl.uniform1f(gl.getUniformLocation(program, 'u_brightness'), p.brightness);
    gl.uniform1f(gl.getUniformLocation(program, 'u_contrast'), p.contrast);
    gl.uniform1f(gl.getUniformLocation(program, 'u_saturation'), p.saturation);
    gl.uniform1i(gl.getUniformLocation(program, 'u_tex'), 0);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    return canvas;
  } catch (err) {
    console.warn('WebGL apply не удался, переходим на CPU-фоллбэк:', err);
    return null;
  }
}
