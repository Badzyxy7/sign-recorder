/* ============================================================
   Sign Clip Recorder - script.js  (PC + mobile, always landscape)
   ============================================================ */

/* ============================ SETTINGS (edit me) ============================ */
// Dropdown categories. Each word/letter gets one entry per hand under a heading.
const CATEGORIES = [
  { name: "WORDS",    labels: ["BATHROOM", "FATHER", "FOOD", "FRIEND", "HOME",
                               "MOTHER", "SCHOOL", "STUDENT", "TEACHER", "WATER"] },
  { name: "ALPHABET", labels: ["J", "Z"] },
];
const SIDES  = ["LEFT", "RIGHT"];
const USE_SUBFOLDERS = true;    // true: save into <chosen folder>/<WORD>/<WORD>_<SIDE>_001.mp4 (creates the subfolder if missing)
                                // false: save everything directly in the chosen folder
const LANDMARK_FORMAT = "npy";  // second file saved next to each video: "npy" (NumPy), "json", or "none"
const MOVE_THRESHOLD = 0.008;   // avg landmark shift per frame (0-1 image units); below this = "still"
const STILL_MS       = 1000;    // stillness duration (after movement) that ends the clip
const COUNTDOWN_S    = 3;       // countdown seconds before recording
const MAX_RECORD_MS  = 10000;   // hard limit on clip length
const NO_HANDS_MS    = 5000;    // cancel if no hands appear within this time of recording start
const MAX_HANDS      = 2;
const REC_W = 1280, REC_H = 720; // every recorded clip is this landscape size (16:9)
const VIDEO_BITRATE  = 4000000; // recording quality (bits per second)
const MP_VERSION     = "0.10.14"; // pinned @mediapipe/tasks-vision version
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MOBILE_ROOT_FOLDER = "clips"; // folder created inside the phone browser's storage

// Preferred recording formats, tried in order. Falls back to WebM if MP4 isn't supported.
const FORMATS = [
  { mime: "video/mp4;codecs=avc1", ext: "mp4" },
  { mime: "video/mp4",             ext: "mp4" },
  { mime: "video/webm;codecs=vp9", ext: "webm" },
  { mime: "video/webm;codecs=vp8", ext: "webm" },
  { mime: "video/webm",            ext: "webm" },
];

/* ============================ ELEMENTS & STATE ============================== */
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const recordBtn = $("recordBtn"), folderBtn = $("folderBtn"), labelSelect = $("label");
const exportBtn = $("exportBtn"), clearBtn = $("clearBtn"), cancelBtn = $("cancelBtn");

// Hidden landscape canvas: the camera image is cropped to 16:9 and drawn here every frame.
// Hand detection and the recording both use THIS canvas, so clips are always landscape.
const recCanvas = document.createElement("canvas");
recCanvas.width = REC_W; recCanvas.height = REC_H;
const recCtx = recCanvas.getContext("2d");
canvas.width = REC_W; canvas.height = REC_H;   // landmark overlay has the same size

// Folder modes:
//  - PC Chrome/Edge:  real folder picker (showDirectoryPicker)
//  - Phones/others:   a "clips" folder created inside the browser storage (OPFS), exported as a ZIP
const hasFSA  = "showDirectoryPicker" in window;
const hasOPFS = !!(navigator.storage && navigator.storage.getDirectory);
const useOPFS = !hasFSA && hasOPFS;
const folderMode = hasFSA || hasOPFS;          // a folder must be created/chosen before recording
const isTouch = window.matchMedia && matchMedia("(pointer: coarse)").matches;

let state = "loading";           // loading | idle | countdown | recording | saving
let stream = null, recStream = null, landmarker = null, dirHandle = null;
let recorder = null, chunks = [], discard = false;
let countdownTimer = null, maxTimer = null, noHandsTimer = null;
let recStart = 0, lastMove = 0, hasMoved = false, handsSeen = false;
let prevHands = null, lastVideoTime = -1, savedCount = 0, fileExt = "mp4";
let landmarkFrames = [];         // one Float32Array(126) per processed frame while recording
const sessionCounts = {};        // fallback numbering when no folder is used

// Build the grouped dropdown: category header (disabled) > word (optgroup) > "<WORD> — LEFT HAND" / "<WORD> — RIGHT HAND"
const ENTRIES = [];
CATEGORIES.forEach(cat => {
  const head = new Option(`── ${cat.name} ──`, "");
  head.disabled = true; head.className = "cat";
  labelSelect.add(head);
  cat.labels.forEach(word => {
    const group = document.createElement("optgroup");
    group.label = word;
    SIDES.forEach(side => {
      const entry = { word, side, prefix: `${word}_${side}` };
      ENTRIES.push(entry);
      group.appendChild(new Option(`${word} — ${side} HAND`, entry.prefix));
    });
    labelSelect.appendChild(group);
  });
});
labelSelect.value = ENTRIES[0].prefix;
const currentEntry = () => ENTRIES.find(e => e.prefix === labelSelect.value);

/* ============================ UI HELPERS ==================================== */
const setStatus = t => { $("status").textContent = t; };
function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

function canRecord() {
  return state === "idle" && stream && landmarker && (dirHandle || !folderMode);
}
function setState(s, msg) {
  state = s;
  if (msg) setStatus(msg);
  recordBtn.disabled = !canRecord();
  folderBtn.disabled = !folderMode || s !== "idle";
  labelSelect.disabled = s !== "idle";
  exportBtn.disabled = !dirHandle || s !== "idle";
  clearBtn.disabled = !dirHandle || s !== "idle";
  cancelBtn.hidden = !(useOPFS && (s === "countdown" || s === "recording"));
}

// Phone held upright: show the preview as a wide 16:9 frame (same crop that gets recorded).
// The old "rotate your phone" lock is no longer needed.
function updateOrientation() {
  const portrait = video.videoWidth > 0 && video.videoWidth < video.videoHeight;
  document.querySelector(".stage").classList.toggle("crop169", portrait);
  $("rotate").hidden = true;
  recordBtn.disabled = !canRecord();
}

/* ============================ CAMERA ======================================== */
async function startCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width:  { ideal: isTouch ? 1920 : 1280 },
      height: { ideal: isTouch ? 1080 : 720 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  lastVideoTime = -1; prevHands = null;
}

// Draw the camera image into the landscape canvas (center crop to 16:9, never mirrored)
function drawLandscapeFrame() {
  const sw = video.videoWidth, sh = video.videoHeight;
  if (!sw || !sh) return;
  const target = REC_W / REC_H;
  let cw = sw, ch = sh;
  if (sw / sh > target) cw = sh * target;      // camera is wider than 16:9: trim the sides
  else ch = sw / target;                        // camera is taller (phone upright): trim top and bottom
  const sx = (sw - cw) / 2, sy = (sh - ch) / 2;
  recCtx.drawImage(video, sx, sy, cw, ch, 0, 0, REC_W, REC_H);
}

video.addEventListener("resize", updateOrientation);
window.addEventListener("orientationchange", () => setTimeout(updateOrientation, 300));

/* ============================ STARTUP ======================================= */
async function init() {
  // Browser support check (Chrome / Edge / Safari needed for MediaRecorder)
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !recCanvas.captureStream) {
    showError("This browser is not supported. Open the page in Chrome (Android / PC), Edge, or Safari (iPhone).");
    return setStatus("Unsupported browser");
  }
  if (useOPFS) {
    folderBtn.textContent = "📁 Create clips folder";
    $("folderName").textContent = "No folder created yet. Tap “Create clips folder” first.";
    $("mobileTools").hidden = false;
  } else if (!hasFSA) {
    showError("Folder saving is not available in this browser. Clips will download normally. Use Chrome or Edge for automatic saving.");
    folderBtn.disabled = true;
    $("folderName").textContent = "Using normal downloads.";
  }
  // Camera
  try {
    await startCamera();
  } catch (err) {
    showError("Camera unavailable: " + (err.name === "NotAllowedError"
      ? "access was blocked. Allow the camera in the address bar and reload."
      : err.message));
    return setStatus("No camera");
  }
  // Hand landmarker (MediaPipe, loaded from jsDelivr)
  try {
    setStatus("Loading hand tracking…");
    const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
    const mp = await import(`${base}/vision_bundle.mjs`);
    const fileset = await mp.FilesetResolver.forVisionTasks(`${base}/wasm`);
    landmarker = await mp.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: "VIDEO", numHands: MAX_HANDS,
    });
    window._mp = mp;
  } catch (err) {
    showError("Could not load hand tracking (check your internet connection): " + err.message);
    return setStatus("Hand tracking failed");
  }
  setState("idle", dirHandle || !folderMode ? "Ready" : (useOPFS ? "Create the clips folder to begin" : "Choose a save folder to begin"));
  updateOrientation();
  refreshNext();
  requestAnimationFrame(loop);
}

/* ============================ FOLDER PICKER / CREATOR ======================= */
folderBtn.addEventListener("click", async () => {
  try {
    if (useOPFS) {
      // Phone: create (or reopen) the "clips" folder inside the browser's private storage
      const root = await navigator.storage.getDirectory();
      dirHandle = await root.getDirectoryHandle(MOBILE_ROOT_FOLDER, { create: true });
      try { await navigator.storage.persist(); } catch {}
      $("folderName").textContent = `Folder “${MOBILE_ROOT_FOLDER}” is ready. Clips are stored inside it.`;
    } else {
      dirHandle = await showDirectoryPicker({ mode: "readwrite" });
      $("folderName").textContent = "Saving to folder: " + dirHandle.name;
    }
    setState("idle", "Ready");
    updateOrientation();
    refreshNext();
  } catch (err) {
    if (err.name !== "AbortError") showError("Could not open folder: " + err.message);
  }
});

/* ============================ LABEL & NUMBERING ============================= */
// Folder that holds this word's clips (one subfolder per word, shared by LEFT and RIGHT)
async function labelDir(word, create) {
  if (!dirHandle) return null;
  if (!USE_SUBFOLDERS) return dirHandle;
  if (!create) {
    try { return await dirHandle.getDirectoryHandle(word); } catch { return null; } // not created yet
  }
  return dirHandle.getDirectoryHandle(word, { create: true });
}

// Next free number for a word + side: continues from existing files, never overwrites
async function nextNumber(entry) {
  let max = sessionCounts[entry.prefix] || 0;
  const dir = await labelDir(entry.word, false);
  if (dir) {
    const re = new RegExp("^" + entry.prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "_(\\d+)\\.(mp4|webm)$", "i");
    for await (const [fname] of dir.entries()) {
      const m = fname.match(re); if (m) max = Math.max(max, parseInt(m[1], 10));
    }
  }
  return max + 1;
}

// Update the "Now signing" card (word + hand badge) and show the next filename
async function refreshNext() {
  const entry = currentEntry();
  $("signWord").textContent = entry.word;
  const badge = $("handBadge");
  badge.textContent = `${entry.side} HAND`;
  badge.dataset.side = entry.side;
  const n = await nextNumber(entry);
  const base = `${entry.prefix}_${String(n).padStart(3, "0")}`;
  const data = LANDMARK_FORMAT === "none" ? "" : ` + ${base}.${LANDMARK_FORMAT}`;
  $("nextFile").textContent = `Next clip: ${base}.${fileExt}${data}`;
}

labelSelect.addEventListener("change", () => { labelSelect.blur(); refreshNext(); }); // blur so Space starts recording

/* ============================ LANDMARK DATA ================================= */
// One frame -> Float32Array(2 hands x 21 landmarks x [x, y, z]) = 126 values.
// Slot 0 = hand MediaPipe labels "Left", slot 1 = "Right". Missing hands stay 0.
function frameVector(res) {
  const out = new Float32Array(2 * 21 * 3);
  const used = [false, false];
  (res.landmarks || []).forEach((hand, i) => {
    const label = res.handedness?.[i]?.[0]?.categoryName;
    let slot = label === "Right" ? 1 : 0;
    if (used[slot]) slot = 1 - slot;
    used[slot] = true;
    hand.forEach((p, j) => {
      const o = (slot * 21 + j) * 3;
      out[o] = p.x; out[o + 1] = p.y; out[o + 2] = p.z;
    });
  });
  return out;
}

// Build a NumPy .npy file (float32, shape [frames, 2, 21, 3]) that np.load() can read
function makeNpy(frames) {
  const n = frames.length;
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${n}, 2, 21, 3), }`;
  header += " ".repeat((64 - ((10 + header.length + 1) % 64)) % 64) + "\n"; // pad to 64-byte boundary
  const hb = new TextEncoder().encode(header);
  const buf = new ArrayBuffer(10 + hb.length + n * 126 * 4);
  const dv = new DataView(buf);
  [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0].forEach((b, i) => dv.setUint8(i, b)); // magic + version 1.0
  dv.setUint16(8, hb.length, true);
  new Uint8Array(buf, 10, hb.length).set(hb);
  const f = new Float32Array(buf, 10 + hb.length, n * 126);
  frames.forEach((v, i) => f.set(v, i * 126));
  return new Blob([buf], { type: "application/octet-stream" });
}

function makeJson(frames) {
  return new Blob([JSON.stringify({ shape: [frames.length, 2, 21, 3], data: frames.map(v => Array.from(v)) })],
                  { type: "application/json" });
}

/* ============================ DETECTION LOOP ================================ */
// Average normalized distance moved by all landmarks since the previous frame.
function measureMovement(hands) {
  let m = 0;
  if (prevHands && prevHands.length === hands.length && hands.length) {
    let sum = 0, n = 0;
    hands.forEach((h, i) => h.forEach((p, j) => {
      sum += Math.hypot(p.x - prevHands[i][j].x, p.y - prevHands[i][j].y); n++;
    }));
    m = sum / n;
  } else if (hands.length && (!prevHands || prevHands.length !== hands.length)) {
    m = MOVE_THRESHOLD * 2;      // a hand entering or a hand count change counts as movement
  }
  prevHands = hands;
  return m;
}

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  drawLandscapeFrame();                                  // landscape crop (this is what gets recorded)
  const res = landmarker.detectForVideo(recCanvas, performance.now());
  const hands = res.landmarks || [];

  // Draw landmarks on the overlay only (the recording is the clean landscape canvas)
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const draw = new _mp.DrawingUtils(ctx);
  for (const h of hands) {
    draw.drawConnectors(h, _mp.HandLandmarker.HAND_CONNECTIONS, { color: "#ffffff", lineWidth: 3 });
    draw.drawLandmarks(h, { color: "#ffffff", fillColor: "#b3122b", lineWidth: 1, radius: 4 });
  }

  const move = measureMovement(hands);
  if (state !== "recording") return;

  landmarkFrames.push(frameVector(res));     // store this frame's landmarks for the data file

  // Auto-stop logic while recording
  const now = performance.now();
  if (hands.length) { handsSeen = true; clearTimeout(noHandsTimer); }
  if (move > MOVE_THRESHOLD) { hasMoved = true; lastMove = now; }
  else if (hasMoved && now - lastMove >= STILL_MS) stopRecording();
}

/* ============================ RECORD FLOW =================================== */
async function startRecording() {
  if (!canRecord()) return;
  showError("");
  setState("countdown");
  for (let n = COUNTDOWN_S; n > 0; n--) {
    $("countdown").textContent = n;
    $("countdown").classList.add("on");
    setStatus("Countdown: " + n);
    await new Promise(r => { countdownTimer = setTimeout(r, 1000); });
    if (state !== "countdown") return;           // cancelled with Esc / Cancel button
  }
  $("countdown").classList.remove("on");

  // Record the landscape canvas (always 16:9, unmirrored, no landmarks), MP4 if supported
  chunks = []; landmarkFrames = []; discard = false; hasMoved = false; handsSeen = false; prevHands = null;
  const fmt = FORMATS.find(f => MediaRecorder.isTypeSupported(f.mime));
  fileExt = fmt ? fmt.ext : "webm";
  recStream = recCanvas.captureStream(30);
  const opts = { videoBitsPerSecond: VIDEO_BITRATE };
  if (fmt) opts.mimeType = fmt.mime;
  recorder = new MediaRecorder(recStream, opts);
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = onRecorderStop;
  recorder.start();
  recStart = lastMove = performance.now();
  setState("recording", "Recording");
  $("badge").classList.add("on");

  maxTimer = setTimeout(stopRecording, MAX_RECORD_MS);
  noHandsTimer = setTimeout(() => {
    if (!handsSeen) { discard = true; setStatus("No hands detected. Clip cancelled."); stopRecording(); }
  }, NO_HANDS_MS);
}

function stopRecording() {
  if (state !== "recording" || !recorder || recorder.state === "inactive") return;
  clearTimers();
  $("badge").classList.remove("on");
  setState("saving", discard ? undefined : "Saving…");
  recorder.stop();
}

function cancelAll() {
  if (state === "countdown") {
    clearTimers(); $("countdown").classList.remove("on");
    setState("idle", "Cancelled");
  } else if (state === "recording") {
    discard = true; setStatus("Cancelled"); stopRecording();
  }
}

function clearTimers() { clearTimeout(countdownTimer); clearTimeout(maxTimer); clearTimeout(noHandsTimer); }

/* ============================ SAVING ======================================== */
async function onRecorderStop() {
  const blob = new Blob(chunks, { type: recorder.mimeType });
  const frames = landmarkFrames;
  const entry = currentEntry();
  const wasDiscarded = discard;
  recStream?.getTracks().forEach(t => t.stop()); recStream = null;
  recorder = null; chunks = []; landmarkFrames = []; discard = false;   // clean up recorder state
  if (wasDiscarded) { setState("idle"); refreshNext(); return; }
  try {
    let dataBlob = null;
    if (LANDMARK_FORMAT === "npy") dataBlob = makeNpy(frames);
    else if (LANDMARK_FORMAT === "json") dataBlob = makeJson(frames);
    const saved = await saveClip(blob, dataBlob, entry);
    savedCount++;
    $("count").textContent = savedCount;
    const li = document.createElement("li"); li.textContent = saved;
    const list = $("clips"); list.querySelector(".empty")?.remove(); list.prepend(li);
    setState("idle", "Saved as " + saved);
  } catch (err) {
    showError("Could not save the clip: " + err.message);
    setState("idle", "Save failed");
  }
  refreshNext();
}

// Fallback writer for browsers (iPhone Safari) that cannot use createWritable():
// a small background worker writes the file with a sync access handle.
let writeWorker = null;
function workerWrite(dir, name, blob) {
  return new Promise(async (resolve, reject) => {
    if (!writeWorker) {
      const code = `
        self.onmessage = async e => {
          const { dir, name, buf, id } = e.data;
          try {
            const fh = await dir.getFileHandle(name, { create: true });
            const h = await fh.createSyncAccessHandle();
            h.truncate(0);
            h.write(new Uint8Array(buf), { at: 0 });
            h.flush(); h.close();
            self.postMessage({ id, ok: true });
          } catch (err) { self.postMessage({ id, ok: false, error: String(err) }); }
        };`;
      writeWorker = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
    }
    const id = Math.random();
    const buf = await blob.arrayBuffer();
    const onMsg = e => {
      if (e.data.id !== id) return;
      writeWorker.removeEventListener("message", onMsg);
      e.data.ok ? resolve() : reject(new Error(e.data.error));
    };
    writeWorker.addEventListener("message", onMsg);
    writeWorker.postMessage({ dir, name, buf, id }, [buf]);
  });
}

// Write one file into a folder handle, or download it if no folder is available
async function writeFile(dir, name, blob) {
  if (dir) {
    const fh = await dir.getFileHandle(name, { create: true });
    if (typeof fh.createWritable === "function") {
      const w = await fh.createWritable();
      await w.write(blob); await w.close();
    } else {
      await workerWrite(dir, name, blob);
    }
  } else {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
}

// Save the video and the landmark file with the same name, e.g. Z_LEFT_003.mp4 + Z_LEFT_003.npy
async function saveClip(videoBlob, dataBlob, entry) {
  const num = await nextNumber(entry);
  sessionCounts[entry.prefix] = num;
  const base = `${entry.prefix}_${String(num).padStart(3, "0")}`;
  const dir = dirHandle ? await labelDir(entry.word, true) : null;   // creates the word subfolder if needed

  await writeFile(dir, `${base}.${fileExt}`, videoBlob);
  if (dataBlob) await writeFile(dir, `${base}.${LANDMARK_FORMAT}`, dataBlob);
  return dataBlob ? `${base}.${fileExt} + ${base}.${LANDMARK_FORMAT}` : `${base}.${fileExt}`;
}

/* ============================ MOBILE: EXPORT AS ZIP ========================= */
// Minimal ZIP writer (no compression, keeps the folder structure clips/WORD/file)
let crcTable = null;
function crc32(u8) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = crcTable[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function makeZip(entries) {               // entries: [{ path, data: Uint8Array }]
  const enc = new TextEncoder();
  const d = new Date();
  const dosTime = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const dosDate = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  const parts = [], central = [];
  let offset = 0;
  for (const e of entries) {
    const name = enc.encode(e.path);
    const crc = crc32(e.data), size = e.data.length;
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true);
    lh.setUint16(8, 0, true);  lh.setUint16(10, dosTime, true); lh.setUint16(12, dosDate, true);
    lh.setUint32(14, crc, true); lh.setUint32(18, size, true); lh.setUint32(22, size, true);
    lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
    parts.push(lh.buffer, name, e.data);

    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true);
    ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true);
    ch.setUint16(12, dosTime, true); ch.setUint16(14, dosDate, true);
    ch.setUint32(16, crc, true); ch.setUint32(20, size, true); ch.setUint32(24, size, true);
    ch.setUint16(28, name.length, true); ch.setUint32(42, offset, true);
    central.push(ch.buffer, name);
    offset += 30 + name.length + size;
  }
  const cdSize = central.reduce((s, p) => s + (p.byteLength ?? p.length), 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, entries.length, true); end.setUint16(10, entries.length, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: "application/zip" });
}

async function collectFiles(dir, prefix, out) {
  for await (const [name, handle] of dir.entries()) {
    if (handle.kind === "file") out.push({ path: prefix + name, handle });
    else await collectFiles(handle, prefix + name + "/", out);
  }
}

exportBtn.addEventListener("click", async () => {
  if (!dirHandle) return;
  try {
    setStatus("Preparing ZIP…");
    exportBtn.disabled = true;
    const list = [];
    await collectFiles(dirHandle, dirHandle.name + "/", list);
    if (!list.length) { setStatus("No clips to save yet."); return; }
    const entries = [];
    for (const f of list) {
      const file = await f.handle.getFile();
      entries.push({ path: f.path, data: new Uint8Array(await file.arrayBuffer()) });
    }
    const zip = makeZip(entries);
    const t = new Date(), p = n => String(n).padStart(2, "0");
    const name = `clips_${t.getFullYear()}${p(t.getMonth() + 1)}${p(t.getDate())}_${p(t.getHours())}${p(t.getMinutes())}.zip`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(zip); a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    setStatus(`ZIP saved to your Downloads (${list.length} files). Upload it to Google Drive.`);
  } catch (err) {
    showError("Could not create the ZIP: " + err.message);
    setStatus("ZIP failed");
  } finally {
    exportBtn.disabled = !dirHandle || state !== "idle";
  }
});

clearBtn.addEventListener("click", async () => {
  if (!dirHandle) return;
  if (!confirm("Delete ALL clips stored in the app? Make sure you already saved the ZIP and uploaded it.")) return;
  try {
    const names = [];
    for await (const [name] of dirHandle.entries()) names.push(name);
    for (const n of names) await dirHandle.removeEntry(n, { recursive: true });
    Object.keys(sessionCounts).forEach(k => delete sessionCounts[k]);
    setStatus("Stored clips cleared.");
    refreshNext();
  } catch (err) {
    showError("Could not clear clips: " + err.message);
  }
});

cancelBtn.addEventListener("click", cancelAll);

/* ============================ FIRST-TIME INSTRUCTIONS ======================= */
const INTRO_KEY = "signRecorderIntroDone";
const INTRO_STEPS = useOPFS ? [
  {
    icon: "📱",
    title: "Hold your phone steady",
    text: "You can hold your phone upright or sideways. Clips are always recorded in landscape (wide) format, so the preview shows the wide frame. Keep your hands inside the preview. Sideways gives the sharpest video.",
  },
  {
    icon: "📁",
    title: "Create the clips folder",
    text: "Tap the “📁 Create clips folder” button. The Record button stays locked until the folder is created. The folder is kept inside this website's storage on your phone.",
    tree: "clips/",
  },
  {
    icon: "🔤",
    title: "Select a word from the dropdown",
    text: "Open the dropdown and pick a word or letter. Each option shows which hand to use, for example BATHROOM — LEFT HAND. The system automatically creates a folder with that word's name inside clips.",
    tree: "clips/\n└─ BATHROOM/   ← created automatically",
  },
  {
    icon: "🎥",
    title: "Record your sign",
    text: `Tap Record. After the ${COUNTDOWN_S}-second countdown, sign with the hand shown in the badge. The recording stops by itself when you hold still. Tap Cancel to discard a clip (nothing is saved). Every clip is saved automatically as 2 files with the same name:`,
    tree: `clips/\n└─ BATHROOM/\n   ├─ BATHROOM_LEFT_001.mp4   (the video)\n   └─ BATHROOM_LEFT_001.${LANDMARK_FORMAT}   (hand landmark data)`,
  },
  {
    icon: "☁️",
    title: "Save the ZIP and upload to Google Drive",
    text: "When you are finished, tap “📦 Save all clips (ZIP)”. The ZIP (with the clips folder inside) is saved to your phone's Downloads. Upload it to our shared Google Drive. Do not rename or edit the files inside.",
  },
] : [
  {
    icon: "📁",
    title: "Create a folder named “clips”",
    text: "On your computer, create a new empty folder and name it clips. Put it somewhere easy to find, like your Desktop or Documents. All your recordings will be saved inside it.",
    tree: "Desktop/\n└─ clips/",
  },
  {
    icon: "📂",
    title: "Choose the folder in the system",
    text: "On this page, click the “📁 Choose save folder” button, select your clips folder, and click Allow when the browser asks for permission. The Record button stays locked until a folder is chosen.",
  },
  {
    icon: "🔤",
    title: "Select a word from the dropdown",
    text: "Open the dropdown and pick a word or letter. Each option shows which hand to use, for example BATHROOM — LEFT HAND. The system automatically creates a folder with that word's name inside clips.",
    tree: "clips/\n└─ BATHROOM/   ← created automatically",
  },
  {
    icon: "🎥",
    title: "Record your sign",
    text: `Click Record (or press Space). After the ${COUNTDOWN_S}-second countdown, sign with the hand shown in the badge. The recording stops by itself when you hold still. Press Esc to cancel a clip (nothing is saved). Every clip is saved automatically as 2 files with the same name:`,
    tree: `clips/\n└─ BATHROOM/\n   ├─ BATHROOM_LEFT_001.mp4   (the video)\n   └─ BATHROOM_LEFT_001.${LANDMARK_FORMAT}   (hand landmark data)`,
  },
  {
    icon: "☁️",
    title: "Upload your clips to Google Drive",
    text: "When you are finished recording, upload your clips folder to our shared Google Drive. Make sure both files of every clip are uploaded, and do not rename or edit them.",
  },
];

const introEl = $("intro");
let introIndex = 0, introOpen = false;

function renderIntro() {
  const step = INTRO_STEPS[introIndex];
  const last = introIndex === INTRO_STEPS.length - 1;
  $("introNum").textContent = introIndex + 1;
  $("introTotal").textContent = INTRO_STEPS.length;
  $("introIcon").textContent = step.icon;
  $("introTitle").textContent = step.title;
  $("introText").textContent = step.text;
  $("introTree").hidden = !step.tree;
  $("introTree").textContent = step.tree || "";
  $("introDots").replaceChildren(...INTRO_STEPS.map((_, i) => {
    const d = document.createElement("span");
    if (i === introIndex) d.className = "on";
    return d;
  }));
  $("introBack").hidden = introIndex === 0;
  $("introNext").textContent = last ? "Got it, start recording" : "Next";
  $("introNext").focus();
}

function openIntro() {
  introIndex = 0; introOpen = true;
  introEl.hidden = false;
  document.querySelector("main").inert = true;       // keeps keyboard focus inside the instructions
  document.body.classList.add("modal-open");
  renderIntro();
}

function closeIntro() {
  introOpen = false;
  introEl.hidden = true;
  document.querySelector("main").inert = false;
  document.body.classList.remove("modal-open");
  try { localStorage.setItem(INTRO_KEY, "1"); } catch {}
}

$("introNext").addEventListener("click", () => {
  if (introIndex < INTRO_STEPS.length - 1) { introIndex++; renderIntro(); }
  else closeIntro();                                  // only way to close: finish the last step
});
$("introBack").addEventListener("click", () => { if (introIndex > 0) { introIndex--; renderIntro(); } });
$("helpBtn").addEventListener("click", openIntro);

let introSeen = false;
try { introSeen = localStorage.getItem(INTRO_KEY) === "1"; } catch {}
if (!introSeen) openIntro();                          // first visit: show automatically

/* ============================ EVENTS ======================================== */
recordBtn.addEventListener("click", startRecording);
document.addEventListener("keydown", e => {
  if (introOpen) return;                              // instructions open: Esc and Space do nothing else
  if (e.code === "Escape") return cancelAll();
  if (e.code === "Space" && e.target.tagName !== "SELECT") {
    e.preventDefault();
    if (!e.repeat) startRecording();
  }
});
window.addEventListener("beforeunload", () => { clearTimers(); stream?.getTracks().forEach(t => t.stop()); });

init();