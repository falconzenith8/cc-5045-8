/* Lightbaker task panel + dependency-free Canvas 3D preview.
   The 3D view rasterizes the actual baked mesh with a z-buffer and samples
   the baked lightmap by UV. It never reads pixels back from a screenshot. */
"use strict";

const $ = (id) => document.getElementById(id);
let currentJob = null;
let pollTimer = null;
let capabilities = null;
let preview = { mesh: null, texture: null, missing: new Set() };

// ------------------------------------------------------------ panel setup
async function loadCapabilities() {
  capabilities = await fetch("/api/capabilities").then((r) => r.json());
  const sceneSel = $("f-scene");
  for (const [name, info] of Object.entries(capabilities.scenes)) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sceneSel.appendChild(opt);
  }
  const atlasSel = $("f-atlas");
  for (const size of capabilities.atlas_choices) {
    const opt = document.createElement("option");
    opt.value = size;
    opt.textContent = size + " x " + size;
    if (size === 256) opt.selected = true;
    atlasSel.appendChild(opt);
  }
  sceneSel.addEventListener("change", updateSceneDesc);
  updateSceneDesc();
}

function updateSceneDesc() {
  const name = $("f-scene").value;
  const info = capabilities.scenes[name];
  $("scene-desc").textContent = info.description + "  ·  单位: " + info.units;
}

function formConfig() {
  const fb = {};
  const failId = parseInt($("fail-id").value, 10);
  const failN = parseInt($("fail-n").value, 10);
  if (failN > 0 && Number.isFinite(failId)) fb[failId] = failN;
  return {
    scene: $("f-scene").value,
    atlas_size: parseInt($("f-atlas").value, 10),
    density: parseFloat($("f-density").value),
    quality: $("f-quality").value,
    samples: parseInt($("f-samples").value, 10) || 0,
    bounces: parseInt($("f-bounces").value, 10),
    job_seed: parseInt($("f-seed").value, 10),
    workers: parseInt($("f-workers").value, 10) || 0,
    fail_blocks: fb,
  };
}

$("bake-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("form-error").textContent = "";
  let resp;
  try {
    resp = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formConfig()),
    });
  } catch (e) {
    $("form-error").textContent = "请求失败: " + e;
    return;
  }
  const data = await resp.json();
  if (!resp.ok) {
    // kernel rejection (e.g. atlas too small) is surfaced here
    $("form-error").textContent = "内核拒绝任务: " + data.error;
    return;
  }
  currentJob = data.job_id;
  $("retry-btn").disabled = false;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshJob, 700);
  refreshJob();
});

$("retry-btn").addEventListener("click", async () => {
  if (!currentJob) return;
  const fb = formConfig().fail_blocks;
  await fetch("/api/jobs/" + currentJob + "/retry", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fail_blocks: fb }),
  });
  refreshJob();
});

// ------------------------------------------------------------ job refresh
async function refreshJob() {
  if (!currentJob) return;
  const data = await fetch("/api/jobs/" + currentJob).then((r) => r.json());
  $("job-id").textContent = currentJob;
  $("job-status").textContent = data.status;
  $("chart-count").textContent = data.chart_count;
  const done = data.blocks.filter((b) => b.state === "done").length;
  const total = data.blocks.filter((b) => b.covered).length;
  const failed = new Set(data.errors.map((e) => e.block_index));
  $("block-done").textContent = done;
  $("block-total").textContent = total;
  $("block-failed").textContent = failed.size;
  renderBlockGrid(data.blocks, failed);
  $("error-log").textContent = data.errors.length
    ? data.errors.map((e) => "块 " + e.block_index + " 失败（第 " +
        e.attempts + " 次尝试）: " + e.error).join("\n")
    : "（无错误）";
  loadResults(data);
  if (data.status === "done" || data.status === "partial") {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderBlockGrid(blocks, failed) {
  const grid = $("block-grid");
  grid.innerHTML = "";
  for (const b of blocks) {
    const cell = document.createElement("div");
    let cls = "empty";
    if (b.covered) {
      if (b.state === "done") cls = "done";
      else if (failed.has(b.index)) cls = "failed";
      else cls = "pending";
    }
    cell.className = "cell " + cls;
    const tip = document.createElement("span");
    tip.className = "tip";
    tip.textContent = "块 " + b.index + " · " + cls +
      (b.attempts ? " · 尝试 " + b.attempts : "");
    cell.appendChild(tip);
    grid.appendChild(cell);
  }
}

// ------------------------------------------------------------ result tabs
let activeImage = "lightmap";
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    activeImage = tab.dataset.img;
    if (currentJob) loadResults(null);
  });
});

async function loadResults(data) {
  if (!data) {
    data = await fetch("/api/jobs/" + currentJob).then((r) => r.json());
  }
  const key = activeImage;
  const url = data.outputs[key] + "?t=" + Date.now();
  $("atlas-img").src = url;
  if (key === "lightmap" || key === "lightmap_clean") {
    try {
      const metrics = await fetch(data.outputs.metrics + "?t=" + Date.now())
        .then((r) => r.json());
      $("metrics").textContent = JSON.stringify(metrics, null, 2);
    } catch (e) { /* still baking */ }
  }
  // feed 3D preview once baking produced mesh + texture
  if (data.status === "done" || data.status === "partial") {
    loadPreview(data);
  }
}

// ------------------------------------------------------------ 3D preview
async function loadPreview(data) {
  const mesh = await fetch(data.outputs.runtime_mesh).then((r) => r.json());
  const img = new Image();
  img.src = data.outputs.lightmap_denoised + "?t=" + Date.now();
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  preview.mesh = mesh;
  preview.texture = { canvas, pixels };
  preview.missing = new Set();
  // magenta texels (1.0, 0, 1.0 after sRGB) mark unfinished regions
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const o = (y * canvas.width + x) * 4;
      if (pixels[o] > 230 && pixels[o + 1] < 30 && pixels[o + 2] > 230) {
        preview.missing.add(y * canvas.width + x);
      }
    }
  }
  $("missing-note").textContent = preview.missing.size
    ? "存在 " + preview.missing.size + " 个未完成纹素，预览中以闪烁洋红标出。"
    : "所有覆盖纹素均已完成烘焙。";
  render3D();
}

const camera = { yaw: 0.7, pitch: 0.45, dist: 13, panX: 0, panY: 0 };
const canvas3d = $("view3d");
const cctx = canvas3d.getContext("2d");

function matMul(a, b) {
  const o = new Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] +
        a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
    }
  }
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0,
    (far + near) / (near - far), -1, 0, 0,
    (2 * far * near) / (near - far), 0];
}
function translate(x, y, z) {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}
function rotateY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
}
function rotateX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}
function applyM(m, v) {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
    m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3],
    m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3],
    m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3],
  ];
}

function sampleTexture(u, v) {
  const t = preview.texture;
  const x = Math.max(0, Math.min(t.canvas.width - 1,
    Math.floor(u * t.canvas.width)));
  const y = Math.max(0, Math.min(t.canvas.height - 1,
    Math.floor((1 - v) * t.canvas.height)));
  const idx = y * t.canvas.width + x;
  const o = idx * 4;
  const p = t.pixels;
  return { rgb: [p[o], p[o + 1], p[o + 2]], missing: preview.missing.has(idx) };
}

function render3D() {
  const W = canvas3d.width, H = canvas3d.height;
  cctx.fillStyle = "#0b0e12";
  cctx.fillRect(0, 0, W, H);
  if (!preview.mesh || !preview.texture) {
    cctx.fillStyle = "#5a6678";
    cctx.font = "14px sans-serif";
    cctx.fillText("完成一次烘焙后在此显示三维预览", 30, H / 2);
    return;
  }
  const mesh = preview.mesh;
  const proj = perspective(Math.PI / 4, W / H, 0.1, 100);
  const view = matMul(
    translate(camera.panX, camera.panY, -camera.dist),
    matMul(rotateX(camera.pitch), rotateY(camera.yaw)));
  const mvp = matMul(proj, view);
  const zbuf = new Float32Array(W * H).fill(Infinity);
  const framebuf = new Uint8ClampedArray(W * H * 4);
  const mode = $("view-mode").value;
  const blink = (Date.now() / 350) % 2 > 1;

  for (const [i0, i1, i2, mid] of mesh.faces) {
    const tri = [mesh.verts[i0], mesh.verts[i1], mesh.verts[i2]];
    const uvs = [mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2]];
    const clipped = tri.map((p) => applyM(mvp, [p[0], p[1], p[2], 1]));
    if (clipped.some((p) => p[3] <= 0.01)) continue;
    const screen = clipped.map((p) => ({
      x: ((p[0] / p[3] + 1) * 0.5) * W,
      y: (1 - (p[1] / p[3] + 1) * 0.5) * H,
      z: p[2] / p[3],
    }));
    const minX = Math.max(0, Math.floor(Math.min(...screen.map((s) => s.x))));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(...screen.map((s) => s.x))));
    const minY = Math.max(0, Math.floor(Math.min(...screen.map((s) => s.y))));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(...screen.map((s) => s.y))));
    const a = screen[0], b = screen[1], c = screen[2];
    const area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (Math.abs(area) < 1e-6) continue;
    const n = mesh.normals[i0];
    const albedo = mesh.materials[mid].albedo.map((v) => v * 255);

    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const cx = px + 0.5, cy = py + 0.5;
        const w0 = ((b.x - cx) * (c.y - cy) - (b.y - cy) * (c.x - cx)) / area;
        const w1 = ((c.x - cx) * (a.y - cy) - (c.y - cy) * (a.x - cx)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
        const z = w0 * a.z + w1 * b.z + w2 * c.z;
        const pi = py * W + px;
        if (z >= zbuf[pi]) continue;
        zbuf[pi] = z;
        const u = w0 * uvs[0][0] + w1 * uvs[1][0] + w2 * uvs[2][0];
        const v = w0 * uvs[0][1] + w1 * uvs[1][1] + w2 * uvs[2][1];
        const tex = sampleTexture(u, v);
        let color;
        if (mode === "albedo") color = albedo;
        else if (mode === "normal") {
          color = [n[0] * 127 + 128, n[1] * 127 + 128, n[2] * 127 + 128];
        } else if (mode === "missing") {
          color = tex.missing ? (blink ? [255, 0, 255] : [90, 0, 90])
            : [60, 60, 66];
        } else {
          color = tex.missing && blink
            ? [255, 0, 255]
            : [tex.rgb[0] * 0.85 + albedo[0] * 0.15,
               tex.rgb[1] * 0.85 + albedo[1] * 0.15,
               tex.rgb[2] * 0.85 + albedo[2] * 0.15];
        }
        const o = pi * 4;
        framebuf[o] = color[0];
        framebuf[o + 1] = color[1];
        framebuf[o + 2] = color[2];
        framebuf[o + 3] = 255;
      }
    }
  }
  cctx.putImageData(new ImageData(framebuf, W, H), 0, 0);
}

// orbit controls
let dragging = false, last = null, panning = false;
canvas3d.addEventListener("mousedown", (e) => {
  dragging = true;
  panning = e.button === 2;
  last = [e.clientX, e.clientY];
});
window.addEventListener("mouseup", () => { dragging = false; });
canvas3d.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - last[0], dy = e.clientY - last[1];
  last = [e.clientX, e.clientY];
  if (panning) {
    camera.panX += dx * 0.01;
    camera.panY -= dy * 0.01;
  } else {
    camera.yaw += dx * 0.008;
    camera.pitch = Math.max(-1.4, Math.min(1.4, camera.pitch + dy * 0.008));
  }
  render3D();
});
canvas3d.addEventListener("wheel", (e) => {
  e.preventDefault();
  camera.dist = Math.max(3, Math.min(40, camera.dist + e.deltaY * 0.01));
  render3D();
}, { passive: false });
$("view-mode").addEventListener("change", render3D);
setInterval(() => { if (preview.mesh) render3D(); }, 400);

loadCapabilities();
