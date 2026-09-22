import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';

const $ = (id) => document.getElementById(id);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#171a19').convertLinearToSRGB();
const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 4000);
const renderer = new THREE.WebGLRenderer({ canvas: $('cloud'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enabled = false;
controls.minDistance = 2;
controls.maxDistance = 1600;
const mapContext = $('map').getContext('2d');
const panoramaContext = $('panorama').getContext('2d');
let data, material, cloudBounds, playing = false, cursor = 0, mode = 'overview', lastTime = 0;
let yaw = 0, pitch = -0.06, dragging = null, lastPanorama = -1;
const atlases = new Map();
const mapImage = new Image();

function icon(button, name, label) {
  button.innerHTML = `<i data-lucide="${name}"></i>`;
  button.title = label;
  button.setAttribute('aria-label', label);
  window.lucide.createIcons();
}

function setPlaying(value) {
  playing = value;
  icon($('play'), value ? 'pause' : 'play', value ? 'Pause' : 'Play');
}

function setMode(value) {
  mode = value;
  controls.enabled = value === 'overview';
  for (const id of ['walk', 'overview']) {
    $(id).classList.toggle('selected', id === value);
    $(id).setAttribute('aria-pressed', String(id === value));
  }
  if (value === 'overview' && cloudBounds) {
    const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
    const distance = cloudBounds.radius / Math.sin(Math.min(vertical, horizontal)) * 1.05;
    controls.target.copy(cloudBounds.center);
    camera.position.copy(cloudBounds.center).add(new THREE.Vector3(0.15, 1, 0.65).normalize().multiplyScalar(distance));
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = true;
  }
}

function currentFrame() {
  const i = Math.floor(cursor), fraction = cursor - i;
  const a = data.frames[i], b = data.frames[Math.min(i + 1, data.frames.length - 1)];
  return {
    position: a.position.map((v, k) => v + (b.position[k] - v) * fraction),
    forward: a.forward.map((v, k) => v + (b.forward[k] - v) * fraction),
  };
}

function drawMap(frame) {
  const [ox, oz] = data.map.offset;
  mapContext.drawImage(mapImage, 0, 0, data.map.width, data.map.height);
  mapContext.beginPath();
  data.frames.forEach((f, i) => mapContext[i ? 'lineTo' : 'moveTo'](f.position[0] + ox, f.position[2] + oz));
  mapContext.strokeStyle = 'rgba(22,112,79,0.72)';
  mapContext.lineWidth = 6;
  mapContext.stroke();
  mapContext.save();
  mapContext.translate(frame.position[0] + ox, frame.position[2] + oz);
  mapContext.rotate(Math.atan2(frame.forward[2], frame.forward[0]) + yaw);
  mapContext.beginPath();
  mapContext.moveTo(23, 0); mapContext.lineTo(-12, -11); mapContext.lineTo(-7, 0); mapContext.lineTo(-12, 11);
  mapContext.closePath();
  mapContext.fillStyle = '#e5473b'; mapContext.strokeStyle = '#fff'; mapContext.lineWidth = 4;
  mapContext.fill(); mapContext.stroke(); mapContext.restore();
}

function drawPanorama() {
  const atlas = data.panoramas;
  const index = Math.floor(cursor / atlas.step);
  if (index === lastPanorama) return;
  const perSheet = atlas.columns * atlas.rows, sheetIndex = Math.floor(index / perSheet);
  if (!atlases.has(sheetIndex)) {
    const img = new Image();
    img.onload = () => { lastPanorama = -1; };
    img.onerror = () => { $('loading').hidden = false; $('loading').textContent = 'Panorama could not load. Please reload.'; };
    img.src = atlas.sheets[sheetIndex];
    atlases.set(sheetIndex, img);
  }
  const img = atlases.get(sheetIndex);
  if (!img.complete || !img.naturalWidth) return;
  const tile = index % perSheet;
  panoramaContext.drawImage(img, (tile % atlas.columns) * atlas.width, Math.floor(tile / atlas.columns) * atlas.height,
    atlas.width, atlas.height, 0, 0, $('panorama').width, $('panorama').height);
  lastPanorama = index;
}

function render(time) {
  requestAnimationFrame(render);
  const dt = Math.min((time - lastTime) / 1000, 0.1);
  lastTime = time;
  if (!data) return;
  if (playing) {
    cursor += dt * Number($('speed').value);
    if (cursor >= data.frames.length - 1) { cursor = data.frames.length - 1; setPlaying(false); }
  }
  const frame = currentFrame();
  if (mode === 'walk') {
    const heading = Math.atan2(frame.forward[2], frame.forward[0]) + yaw;
    // Use the common road-plane convention of the displayed fused cloud.
    camera.position.set(frame.position[0], 2.3, frame.position[2]);
    camera.lookAt(camera.position.x + Math.cos(heading), camera.position.y + Math.tan(pitch), camera.position.z + Math.sin(heading));
  } else controls.update();
  renderer.render(scene, camera);
  drawMap(frame);
  drawPanorama();
  $('timeline').value = Math.floor(cursor);
  $('progress').value = `${Math.floor(cursor) + 1} / ${data.frames.length}`;
}

new ResizeObserver(() => {
  const box = $('stage').getBoundingClientRect();
  renderer.setSize(box.width, box.height, false);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}).observe($('stage'));

$('play').onclick = () => {
  if (cursor >= data.frames.length - 1) cursor = 0;
  if (!playing && mode === 'overview') setMode('walk');
  setPlaying(!playing);
};
$('timeline').oninput = (event) => { cursor = Number(event.target.value); };
$('walk').onclick = () => setMode('walk');
$('overview').onclick = () => setMode('overview');
$('reset').onclick = () => { cursor = 0; yaw = 0; pitch = -0.06; setPlaying(false); setMode(mode); };
$('point-size').oninput = (event) => { if (material) material.size = Number(event.target.value); };
$('fullscreen').onclick = async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await document.documentElement.requestFullscreen();
};
document.addEventListener('fullscreenchange', () => icon($('fullscreen'), document.fullscreenElement ? 'minimize' : 'maximize', document.fullscreenElement ? 'Exit full screen' : 'Full screen'));
$('cloud').addEventListener('pointerdown', (event) => {
  if (mode !== 'walk') return;
  dragging = [event.clientX, event.clientY];
  $('cloud').setPointerCapture(event.pointerId);
});
$('cloud').addEventListener('pointermove', (event) => {
  if (!dragging || mode !== 'walk') return;
  yaw -= (event.clientX - dragging[0]) * 0.004;
  pitch = THREE.MathUtils.clamp(pitch + (event.clientY - dragging[1]) * 0.003, -1.1, 1.1);
  dragging = [event.clientX, event.clientY];
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) $('cloud').addEventListener(event, () => { dragging = null; });

async function unpack(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

async function init() {
  window.lucide.createIcons();
  const response = await fetch('scene.json');
  if (!response.ok) throw new Error('Could not load scene');
  const manifest = await response.json();
  $('loading').textContent = 'Loading 2 million points...';
  const positions = await unpack(manifest.positions);
  const colors = await unpack(manifest.colors);
  mapImage.src = manifest.map.image;
  await mapImage.decode();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(colors), 3, true));
  geometry.computeBoundingSphere();
  cloudBounds = geometry.boundingSphere;
  material = new THREE.PointsMaterial({ size: Number($('point-size').value), sizeAttenuation: false, vertexColors: true });
  // Stored RGB values are already display encoded, as in the source walkthrough.
  material.toneMapped = false;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  scene.add(new THREE.Points(geometry, material));
  data = manifest;
  $('map').width = data.map.width;
  $('map').height = data.map.height;
  $('timeline').max = data.frames.length - 1;
  $('timeline').disabled = false;
  $('play').disabled = false;
  $('point-count').textContent = `${(data.count / 1e6).toFixed(1)}M points`;
  $('loading').hidden = true;
  setMode('overview');
  document.body.dataset.ready = 'true';
  requestAnimationFrame(render);
}

init().catch((error) => {
  $('loading').hidden = false;
  $('loading').textContent = 'The demo could not load. Please reload in a WebGL-enabled browser.';
  console.error(error);
});
