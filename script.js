/* ============================================================
   Sign Clip Recorder - script.js  (desktop / laptop only)
   ============================================================ */

/* ============================ SETTINGS (edit me) ============================ */
// Dropdown categories. Each word/letter gets one entry per hand under a heading.
const CATEGORIES = [
  { name: "WORDS",    labels: ["BATHROOM", "FATHER", "FOOD", "FRIEND", "HOME",
                               "MOTHER", "SCHOOL", "STUDENT", "TEACHER", "WATER"] },
  { name: "ALPHABET", labels: ["J", "Z"] },
  { name: "IDLE (RANDOM SIGNS)", labels: ["IDLE"] },
];
const SIDES  = ["LEFT", "RIGHT"];
const USE_SUBFOLDERS = true;    // true: save into <chosen folder>/<WORD>/<WORD>_<SIDE>_001.mp4 (creates the subfolder if missing)
                                // false: save everything directly in the chosen folder
const LANDMARK_FORMAT = "npy";  // second file saved next to each video: "npy" (NumPy), "json", or "none"

// When does a clip end by itself?
const STOP_MODE      = "hands-out"; // "hands-out": when your hands leave the frame
                                    // "still": when your hands hold still after moving
const HANDS_OUT_MS   = 1000;    // hands-out mode: how long the hands must be out of frame
const MOVE_THRESHOLD = 0.008;   // still mode: avg landmark shift per frame (0-1); below this = "still"
const STILL_MS       = 1000;    // still mode: stillness duration (after movement) that ends the clip

const COUNTDOWN_S    = 3;       // countdown seconds before recording
const MAX_RECORD_MS  = 10000;   // hard limit on clip length
const NO_HANDS_MS    = 5000;    // cancel if no hands appear within this time of recording start
const MAX_HANDS      = 2;

const REC_W = 1280, REC_H = 720;      // every recorded clip is this landscape size (16:9)
const DETECT_W = 640, DETECT_H = 360; // hand detection runs on this smaller image (much faster)
const REC_FPS  = 30;                  // recording frame rate
const VIDEO_BITRATE  = 4000000;       // recording quality (bits per second)

const MP_VERSION     = "0.10.14";     // pinned @mediapipe/tasks-vision version
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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
const countdownEl = $("countdown"), badgeEl = $("badge");

canvas.width = DETECT_W; canvas.height = DETECT_H;   // landmark overlay (small = fast to draw)

// Small canvas used only for hand detection (always a 16:9 crop of the camera)
const detCanvas = document.createElement("canvas");
detCanvas.width = DETECT_W; detCanvas.height = DETECT_H;
const detCtx = detCanvas.getContext("2d");

// Big canvas used for recording ONLY when the webcam is not 16:9 (e.g. a 4:3 webcam).
// A normal 16:9 webcam is recorded directly from the camera stream, which is the smoothest option.
const recCanvas = document.createElement("canvas");
recCanvas.width = REC_W; recCanvas.height = REC_H;
const recCtx = recCanvas.getContext("2d");
let recordFromCanvas = false;

// Desktop / laptop only: phones and tablets are blocked.
const isMobile = /Android|iPhone|iPad|iPod|Mobile|Silk/i.test(navigator.userAgent)
  || (/Macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1);   // iPad that pretends to be a Mac
const hasFSA = "showDirectoryPicker" in window;   // real folder picker (Chrome / Edge on desktop)

let state = "loading";           // loading | idle | countdown | recording | saving
let stream = null, recStream = null, landmarker = null, dirHandle = null;
let recorder = null, chunks = [], discard = false;
let countdownTimer = null, maxTimer = null, noHandsTimer = null;
let lastMove = 0, hasMoved = false, handsSeen = false, lastHandSeen = 0;
let prevHands = null, lastVideoTime = -1, savedCount = 0, fileExt = "mp4";
let landmarkFrames = [];         // one Float32Array(126) per processed frame while recording
let drawer = null;               // MediaPipe drawing helper, created once
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
  return state === "idle" && stream && landmarker && (dirHandle || !hasFSA);
}
function setState(s, msg) {
  state = s;
  if (msg) setStatus(msg);
  recordBtn.disabled = !canRecord();
  folderBtn.disabled = !hasFSA || s !== "idle";
  labelSelect.disabled = s !== "idle";
}

/* ============================ CAMERA ======================================== */
async function startCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width:  { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: REC_FPS, max: REC_FPS },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  lastVideoTime = -1; prevHands = null;
}

// Center-crop the camera image to 16:9 (never mirrored).
// Small copy -> hand detection. Big copy -> recording, only if the webcam is not already 16:9.
function drawLandscapeFrame() {
  const sw = video.videoWidth, sh = video.videoHeight;
  if (!sw || !sh) return;
  const target = REC_W / REC_H;
  let cw = sw, ch = sh;
  if (sw / sh > target) cw = sh * target;      // camera is wider than 16:9: trim the sides
  else ch = sw / target;                        // camera is taller (4:3): trim top and bottom
  const sx = (sw - cw) / 2, sy = (sh - ch) / 2;

  detCtx.drawImage(video, sx, sy, cw, ch, 0, 0, DETECT_W, DETECT_H);
  recordFromCanvas = Math.abs(sw / sh - target) > 0.02;
  if (recordFromCanvas) recCtx.drawImage(video, sx, sy, cw, ch, 0, 0, REC_W, REC_H);
}

/* ============================ STARTUP ======================================= */
async function init() {
  // Phones and tablets are not supported
  if (isMobile) {
    showError("This system works on a laptop or desktop computer only. Please open it on a computer using Chrome or Edge.");
    recordBtn.disabled = true; folderBtn.disabled = true; labelSelect.disabled = true;
    return setStatus("Desktop only");
  }
  // Browser support check (Chrome / Edge recommended)
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !recCanvas.captureStream) {
    showError("This browser is not supported. Please use Chrome or Edge on your computer.");
    return setStatus("Unsupported browser");
  }
  if (!hasFSA) {
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
  // Hand landmarker (MediaPipe, loaded from jsDelivr). GPU first, CPU as a fallback.
  try {
    setStatus("Loading hand tracking…");
    const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
    const mp = await import(`${base}/vision_bundle.mjs`);
    const fileset = await mp.FilesetResolver.forVisionTasks(`${base}/wasm`);
    const makeLandmarker = delegate => mp.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO", numHands: MAX_HANDS,
    });
    try { landmarker = await makeLandmarker("GPU"); }
    catch { landmarker = await makeLandmarker("CPU"); }
    window._mp = mp;
  } catch (err) {
    showError("Could not load hand tracking (check your internet connection): " + err.message);
    return setStatus("Hand tracking failed");
  }
  setState("idle", dirHandle || !hasFSA ? "Ready" : "Choose a save folder to begin");
  refreshNext();
  scheduleNext();
}

/* ============================ FOLDER PICKER ================================= */
folderBtn.addEventListener("click", async () => {
  try {
    dirHandle = await showDirectoryPicker({ mode: "readwrite" });
    $("folderName").textContent = "Saving to folder: " + dirHandle.name;
    setState("idle", "Ready");
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
// Average normalized distance moved by all landmarks since the previous frame (still mode only).
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

// Run once per camera frame (requestVideoFrameCallback), falls back to requestAnimationFrame
function scheduleNext() {
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

function loop() {
  scheduleNext();
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  drawLandscapeFrame();
  const res = landmarker.detectForVideo(detCanvas, performance.now());   // detect on the small image
  const hands = res.landmarks || [];

  // Draw landmarks on the overlay only (the recording never contains them)
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawer ||= new _mp.DrawingUtils(ctx);
  for (const h of hands) {
    drawer.drawConnectors(h, _mp.HandLandmarker.HAND_CONNECTIONS, { color: "#ffffff", lineWidth: 2 });
    drawer.drawLandmarks(h, { color: "#ffffff", fillColor: "#b3122b", lineWidth: 1, radius: 3 });
  }

  if (state !== "recording") return;

  landmarkFrames.push(frameVector(res));     // store this frame's landmarks for the data file

  // Auto-stop logic while recording
  const now = performance.now();
  if (hands.length) { handsSeen = true; lastHandSeen = now; clearTimeout(noHandsTimer); }

  if (STOP_MODE === "hands-out") {
    // End the clip once hands were seen and then have been out of the frame for HANDS_OUT_MS
    if (handsSeen && !hands.length && now - lastHandSeen >= HANDS_OUT_MS) stopRecording();
  } else {
    if (measureMovement(hands) > MOVE_THRESHOLD) { hasMoved = true; lastMove = now; }
    else if (hasMoved && now - lastMove >= STILL_MS) stopRecording();
  }
}

/* ============================ RECORD FLOW =================================== */
async function startRecording() {
  if (!canRecord()) return;
  showError("");
  setState("countdown");
  for (let n = COUNTDOWN_S; n > 0; n--) {
    countdownEl.textContent = n;
    countdownEl.classList.add("on");
    setStatus("Countdown: " + n);
    await new Promise(r => { countdownTimer = setTimeout(r, 1000); });
    if (state !== "countdown") return;           // cancelled with Esc
  }
  countdownEl.classList.remove("on");

  chunks = []; landmarkFrames = []; discard = false; hasMoved = false; handsSeen = false; prevHands = null;
  const fmt = FORMATS.find(f => MediaRecorder.isTypeSupported(f.mime));
  fileExt = fmt ? fmt.ext : "webm";

  // Smooth path: record the webcam stream itself when it is already 16:9 (independent of detection speed).
  // Only non-16:9 webcams go through the cropping canvas.
  recStream = recordFromCanvas ? recCanvas.captureStream(REC_FPS) : stream.clone();
  const opts = { videoBitsPerSecond: VIDEO_BITRATE };
  if (fmt) opts.mimeType = fmt.mime;
  recorder = new MediaRecorder(recStream, opts);
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = onRecorderStop;
  recorder.start();
  lastMove = lastHandSeen = performance.now();
  setState("recording", "Recording");
  badgeEl.classList.add("on");

  maxTimer = setTimeout(stopRecording, MAX_RECORD_MS);
  noHandsTimer = setTimeout(() => {
    if (!handsSeen) { discard = true; setStatus("No hands detected. Clip cancelled."); stopRecording(); }
  }, NO_HANDS_MS);
}

function stopRecording() {
  if (state !== "recording" || !recorder || recorder.state === "inactive") return;
  clearTimers();
  badgeEl.classList.remove("on");
  setState("saving", discard ? undefined : "Saving…");
  recorder.stop();
}

function cancelAll() {
  if (state === "countdown") {
    clearTimers(); countdownEl.classList.remove("on");
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
  recStream?.getTracks().forEach(t => t.stop()); recStream = null;   // stops only the clone, not the live camera
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

// Write one file into a folder handle, or download it if no folder is available
async function writeFile(dir, name, blob) {
  if (dir) {
    const fh = await dir.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.write(blob); await w.close();
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

/* ============================ FIRST-TIME INSTRUCTIONS ======================= */
const INTRO_KEY = "signRecorderIntroDone";
const INTRO_STEPS = [
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
    text: `Click Record (or press Space). After the ${COUNTDOWN_S}-second countdown, sign with the hand shown in the badge. `
        + (STOP_MODE === "hands-out"
            ? "The recording stops by itself when you move your hands out of the frame. "
            : "The recording stops by itself when you hold still. ")
        + "Press Esc to cancel a clip (nothing is saved). Every clip is saved automatically as 2 files with the same name:",
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
if (!introSeen && !isMobile) openIntro();             // first visit on a computer: show automatically

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