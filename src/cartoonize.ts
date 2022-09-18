import * as tf from '@tensorflow/tfjs';
import * as wasm from '@tensorflow/tfjs-backend-wasm';

const options = {
  // modelUrl: '../models/hayao/model.json', // Miyazaki Hayao (heavy outlines and blue tint)
  // modelUrl: '../models/hosoda/model.json', // Hosoda Mamuru (accented horizontal & vertical lines with natural tint)
  // modelUrl: '../models/shinkai/model.json', // Shinkai Makoto (natural without outlines and orange tint)
  // modelUrl: '../models/paprika/model.json', // Paprika (blurred curves with outlines and blue tint) // 2700ms
  // modelUrl: '../models/paprika-light/model.json', // 500ms
  modelUrl: '../models/shinkai-light/model.json', // 450ms
  inputSize: [256, 256],
};

const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('cartoon') as HTMLCanvasElement;

let model;

async function log(...msg) {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  // eslint-disable-next-line no-console
  console.log(ts, ...msg);
  const div = document.getElementById('log') as HTMLElement;
  div.innerText += ts + ': ' + msg.join(' ') + '\n';
  div.scrollTop = div.scrollHeight;
}

async function detect() {
  const t: Record<string, tf.Tensor> = {}; // object to hold all interim tensors so they can be automatically deallocated
  t.pixels = await tf.browser.fromPixelsAsync(video);
  t.resize = tf.image.resizeBilinear(t.pixels as tf.Tensor3D, options.inputSize as [number, number]);
  t.float = tf.cast(t.resize, 'float32');
  t.div = tf.div(t.float, 255);
  t.norm = tf.sub(t.div, 0.5);
  // transpose to BGR and then transpose back
  t.expand = tf.expandDims(t.norm, 0);
  const t0 = performance.now();
  t.cartoon = model.predict(t.expand);
  const t1 = performance.now();
  log('predict:', Math.round(t1 - t0), 'ms');
  t.squeeze = tf.squeeze(t.cartoon);

  t.mul = tf.mul(t.squeeze, 1);
  t.add = tf.add(t.mul, 1);
  t.clipped = tf.clipByValue(t.add, 0, 1) as tf.Tensor3D;
  tf.browser.toPixels(t.clipped as tf.Tensor3D, canvas);

  for (const tensor of Object.keys(t)) tf.dispose(t[tensor]);
  if (!video.paused) requestAnimationFrame(detect);
}

async function initWebCam() {
  if (!navigator.mediaDevices) return null;
  const constraints = {
    audio: false,
    video: { facingMode: 'user', resizeMode: 'none', width: { ideal: 640 } },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  if (stream) video.srcObject = stream;
  else return null;
  window.addEventListener('click', () => {
    if (video.paused) {
      video.play();
      log('play...');
      detect();
    } else {
      log('pause...');
      video.pause();
    }
  });
  return new Promise((resolve) => {
    video.onloadeddata = async () => {
      video.play();
      log('start camera...');
      resolve(video);
    };
  });
}

async function initTFJS() {
  wasm.setWasmPaths('../node_modules/@tensorflow/tfjs-backend-wasm/dist/');
  tf.setBackend('webgl');
  tf.enableProdMode();
  tf.ENV.set('WEBGL_CPU_FORWARD', true);
  tf.ENV.set('WEBGL_PACK_DEPTHWISECONV', false);
  tf.ENV.set('WEBGL_USE_SHAPES_UNIFORMS', true);
  await tf.ready();
  log('tf version:', tf.version_core, 'backend:', tf.getBackend());
  model = await tf.loadGraphModel(options.modelUrl);
  log('model:', model.modelUrl);
  log('memory:', tf.engine().state.numBytes, 'tensors:', tf.engine().state.numTensors);
  const zeros = tf.zeros([1, options.inputSize[0], options.inputSize[1], 3]);
  const t0 = performance.now();
  const warmup = model.predict(zeros);
  const t1 = performance.now();
  log('warmup:', Math.round(t1 - t0), 'ms');
  tf.dispose(zeros);
  tf.dispose(warmup);
}

async function main() {
  await initTFJS();
  await initWebCam();
  detect();
}

window.onload = main;
