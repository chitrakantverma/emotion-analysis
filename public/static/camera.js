'use strict';

const byId = id => document.getElementById(id);
const video = byId('video');
const overlay = byId('overlay');
const overlayContext = overlay.getContext('2d');
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d');
const classes = ['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral'];
const counts = Object.fromEntries(classes.map(label => [label, 0]));
let stream = null;
let generation = 0;
let timer = null;
let requestController = null;
let total = 0;
let recent = [];
let modelReady = false;
let starting = false;
let sessionVersion = 0;

function title(label) { return label.charAt(0).toUpperCase() + label.slice(1); }

function status(message, isError = false) {
  byId('camera-status').textContent = message;
  byId('camera-status').classList.toggle('error', isError);
}

function renderSummary() {
  byId('session-total').textContent = `${total} prediction${total === 1 ? '' : 's'}`;
  byId('counts').replaceChildren(...classes.map(label => {
    const container = document.createElement('div');
    const heading = document.createElement('span');
    const count = document.createElement('strong');
    heading.textContent = title(label);
    count.textContent = counts[label];
    container.append(heading, count);
    return container;
  }));
  const rows = recent.map(item => {
    const row = document.createElement('tr');
    for (const value of [item.time, title(item.expression), `${(item.confidence * 100).toFixed(1)}%`]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  if (!rows.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 3;
    cell.textContent = 'No predictions yet.';
    row.append(cell);
    rows.push(row);
  }
  byId('history').replaceChildren(...rows);
}

function clearResult() {
  for (const id of ['expression', 'confidence', 'response-time', 'face-count']) byId(id).textContent = '—';
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
}

function stopCamera(message = 'Camera stopped.') {
  generation += 1; // Ignore results from the previous camera session.
  clearTimeout(timer);
  requestController?.abort();
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  starting = false;
  byId('camera-placeholder').hidden = false;
  byId('stop').disabled = true;
  byId('start').disabled = !modelReady;
  clearResult();
  status(message);
}

async function startCamera() {
  if (stream || starting || !modelReady) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    status('Camera access requires HTTPS or localhost in a supported browser.', true);
    return;
  }
  starting = true;
  const token = ++generation;
  byId('start').disabled = true;
  byId('stop').disabled = false;
  status('Waiting for camera permission…');
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user'}, audio: false,
    });
    if (token !== generation) {
      acquired.getTracks().forEach(track => track.stop());
      return;
    }
    stream = acquired;
    video.srcObject = acquired;
    await video.play();
    if (token !== generation) return;
    // Align the overlay with the camera frame.
    if (video.videoWidth && video.videoHeight) {
      video.parentElement.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    }
    starting = false;
    byId('camera-placeholder').hidden = true;
    acquired.getVideoTracks()[0].addEventListener('ended', () => {
      if (token === generation) stopCamera('Camera disconnected.');
    });
    status('Camera running.');
    processFrame(token);
  } catch (error) {
    if (token !== generation) return;
    stopCamera();
    const messages = {
      NotAllowedError: 'Camera permission denied. Allow it in browser settings and try again.',
      NotFoundError: 'No camera found. Connect a webcam and try again.',
      NotReadableError: 'Camera is busy. Close other apps using it and try again.',
    };
    status(messages[error.name] || 'Could not start the camera.', true);
  }
}

async function processFrame(token) {
  if (!stream || token !== generation) return;
  const sampleVersion = sessionVersion;
  try {
    if (!video.videoWidth || !video.videoHeight) return;
    const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);
    overlay.width = capture.width;
    overlay.height = capture.height;
    captureContext.drawImage(video, 0, 0, capture.width, capture.height);
    const frame = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.8));
    if (token !== generation || !frame) return;
    requestController = new AbortController();
    const timeout = setTimeout(() => requestController?.abort(), 10000);
    let response;
    try {
      response = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'image/jpeg'},
        body: frame, signal: requestController.signal,
      });
    } finally { clearTimeout(timeout); }
    const result = await response.json();
    if (token !== generation) return;
    if (!response.ok) throw new Error(result.error || 'Prediction request failed.');
    if (sampleVersion !== sessionVersion) return;
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    byId('face-count').textContent = result.face_count;
    byId('response-time').textContent = `${result.processing_ms} ms`;
    if (result.box) {
      const {x, y, width, height} = result.box;
      overlayContext.strokeStyle = '#50d27a';
      overlayContext.lineWidth = 2;
      overlayContext.strokeRect(x, y, width, height);
    }
    if (result.status === 'no_face') {
      byId('expression').textContent = 'No face detected';
      byId('confidence').textContent = '—';
      status('Keep your face within the frame.');
    } else if (result.status === 'out_of_scope') {
      byId('expression').textContent = 'Other expression';
      byId('confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
      status(`${title(result.raw_expression)} detected — outside the six selected classes.`);
    } else if (result.status === 'model_unavailable') {
      stopCamera('Model unavailable. Reload the page to try again.');
    } else if (result.status === 'ok') {
      byId('expression').textContent = title(result.expression);
      byId('confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
      counts[result.expression] += 1;
      total += 1;
      recent.unshift({time: new Date().toLocaleTimeString(), expression: result.expression, confidence: result.confidence});
      recent = recent.slice(0, 10);
      renderSummary();
      status(result.face_count > 1 ? 'Processing the largest face.' : 'Live prediction');
    }
  } catch (error) {
    if (token !== generation) return;
    clearResult();
    status(error.name === 'AbortError' ? 'Request timed out. Retrying…' : error.message, true);
  } finally {
    if (stream && token === generation) timer = setTimeout(() => processFrame(token), 650);
  }
}

byId('start').addEventListener('click', startCamera);
byId('stop').addEventListener('click', () => stopCamera());
byId('reset').addEventListener('click', () => {
  sessionVersion += 1;
  classes.forEach(label => { counts[label] = 0; });
  total = 0;
  recent = [];
  clearResult();
  renderSummary();
  status(stream ? 'Session reset. Camera running.' : 'Session reset.');
});
window.addEventListener('pagehide', () => stopCamera());
renderSummary();
fetch('/health').then(response => {
  if (!response.ok) throw new Error('Server unavailable');
  return response.json();
}).then(state => {
  modelReady = state.ready;
  byId('model-status').textContent = state.ready ? 'Ready' : 'Model unavailable';
  byId('model-status').classList.toggle('error', !state.ready);
  byId('start').disabled = !state.ready;
  if (!state.ready) status(`Model unavailable: ${state.error}`, true);
}).catch(() => {
  byId('model-status').textContent = 'Server unavailable';
  byId('model-status').classList.add('error');
  status('Could not reach the server. Reload the page or try again shortly.', true);
});
