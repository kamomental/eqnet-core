(() => {
  const shell = document.querySelector(".living-shell");
  const canvas = document.getElementById("worldCanvas");
  const ctx = canvas.getContext("2d");
  const worldUrl = shell?.dataset.worldUrl || "/worlds/harbor_town.json";
  const stateUrl = shell?.dataset.stateUrl || "/project-atri/2d-state";
  const eventUrl = shell?.dataset.eventUrl || "/project-atri/2d-event";
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const POLL_MS = 4000;

  const ui = {
    modeLabel: document.getElementById("modeLabel"),
    zoneLabel: document.getElementById("zoneLabel"),
    ambientLabel: document.getElementById("ambientLabel"),
    ambientText: document.getElementById("ambientText"),
    memoryAnchor: document.getElementById("memoryAnchor"),
    memoryText: document.getElementById("memoryText"),
    activityLabel: document.getElementById("activityLabel"),
    presenceText: document.getElementById("presenceText"),
    worldTitle: document.getElementById("worldTitle"),
    worldTypeLabel: document.getElementById("worldTypeLabel"),
    entityList: document.getElementById("entityList"),
    modeDescription: document.getElementById("modeDescription"),
    energyMeter: document.getElementById("energyMeter"),
    stressMeter: document.getElementById("stressMeter"),
    loveMeter: document.getElementById("loveMeter"),
    attentionMeter: document.getElementById("attentionMeter"),
    energyValue: document.getElementById("energyValue"),
    stressValue: document.getElementById("stressValue"),
    loveValue: document.getElementById("loveValue"),
    attentionValue: document.getElementById("attentionValue"),
  };

  const modeButtons = [...document.querySelectorAll(".mode-chip")];
  const modeEventMap = {
    reality: {
      eventType: "rest_exit",
      payload: { zone_id: "market", world_type: "infrastructure", world_source: "surface" },
    },
    streaming: {
      eventType: "stream_stage_enter",
      payload: { zone_id: "stream_stage", world_type: "stage", world_source: "surface" },
    },
    simulation: {
      eventType: "sim_episode_start",
      payload: { zone_id: "trial_quarter", world_type: "community", world_source: "surface" },
    },
  };

  let world = null;
  let runtimeState = null;
  let isPostingEvent = false;
  let pollTimer = null;
  const actor = { x: 10, y: 9, tx: 10, ty: 9, bob: 0 };
  const guide = { x: 18, y: 9, tx: 18, ty: 9, bob: 0 };

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * DPR);
    canvas.height = Math.floor(rect.height * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function prettify(text, fallback = "unknown") {
    const raw = String(text || "").trim();
    if (!raw) return fallback;
    return raw
      .split(/[_-]+/g)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function titleCase(text) {
    const raw = String(text || "").trim().toLowerCase();
    if (!raw) return "Reality";
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  }

  function setMeter(meter, label, value) {
    const numeric = Number.isFinite(value) ? value : 0;
    const pct = `${Math.round(clamp(numeric, 0, 1) * 100)}%`;
    meter.style.width = pct;
    label.textContent = numeric.toFixed(2);
  }

  function renderEntities(entities) {
    ui.entityList.innerHTML = "";
    for (const entity of entities) {
      const item = document.createElement("li");
      item.innerHTML = `<div><strong>${entity.name}</strong><div class="entity-kind">${entity.kind}</div></div><span class="scene-badge">${entity.badge}</span>`;
      ui.entityList.appendChild(item);
    }
  }

  function normalizeEntities(state) {
    const names = state?.social?.nearby_entities || [];
    const entities = names.map((name) => {
      const text = String(name);
      if (text === "user") return { name: "user", kind: "bond anchor", badge: "near" };
      if (text === "world") return { name: "world", kind: "ambient field", badge: "wide" };
      if (text.startsWith("person_")) return { name: text, kind: "nearby person", badge: "active" };
      return { name: text, kind: "scene object", badge: "seen" };
    });
    if (!entities.some((entity) => entity.name === "user")) {
      entities.unshift({ name: "user", kind: "bond anchor", badge: "core" });
    }
    return entities.slice(0, 6);
  }

  function ambientSummary(state) {
    const mode = String(state?.identity?.mode || "reality");
    const timePhase = prettify(state?.world?.time_phase, "Day");
    const weather = prettify(state?.world?.weather, "Clear");
    const stress = Number(state?.body?.stress || 0);
    const retrievalHitCount = Number(state?.memory?.retrieval_hit_count || 0);
    const recallActive = Boolean(state?.activity?.recall_active);
    const voiceLevel = Number(state?.sensing?.voice_level || 0);
    const bodyStress = Number(state?.sensing?.body_stress_index || 0);
    const privacyTags = state?.sensing?.privacy_tags || [];
    const placeId = prettify(state?.sensing?.place_id || state?.world?.zone_id, "Market");

    let label = `${timePhase.toLowerCase()} ${weather.toLowerCase()}`;
    let text = `The lifeform stays in ${placeId} and keeps attention open.`;

    if (mode === "streaming") {
      label = "public shimmer";
      text = "Expression lifts into a public surface while identity stays continuous underneath.";
    } else if (mode === "simulation") {
      label = "accelerated rehearsal";
      text = "The simulation layer is active, but only compact lessons should return to reality.";
    } else if (privacyTags.includes("private")) {
      label = "private hush";
      text = "The place is marked as private, so expression stays softer and more careful.";
    } else if (bodyStress >= 0.72 || stress >= 0.62) {
      label = "guarded pressure";
      text = "Stress is elevated, so the world is being read carefully before action opens up.";
    } else if (voiceLevel >= 0.58) {
      label = "nearby voice";
      text = "A stronger voice signal is present, so attention leans toward live interaction.";
    } else if (recallActive || retrievalHitCount > 0) {
      label = "quiet ignition";
      text = "Recent cues are lighting a short memory chain without letting it swallow the scene.";
    }
    return { label, text };
  }

  function memorySummary(state) {
    const anchor = String(state?.memory?.dominant_anchor || "").trim() || "none";
    const hitCount = Number(state?.memory?.retrieval_hit_count || 0);
    const perception = Boolean(state?.memory?.perception_available);
    let text = "Memory is available, but it is staying in the background for now.";
    if (hitCount > 0 && perception) {
      text = `Vision and recall are both live. ${hitCount} memory cue${hitCount === 1 ? "" : "s"} are near the surface.`;
    } else if (hitCount > 0) {
      text = `${hitCount} memory cue${hitCount === 1 ? "" : "s"} are active around the current scene.`;
    } else if (perception) {
      text = "The current scene is visible and may become a future memory anchor.";
    }
    return { anchor, text };
  }

  function presenceSummary(state) {
    const route = prettify(state?.activity?.route, "Watch");
    const intent = prettify(state?.activity?.intent, "Attend");
    const mode = String(state?.identity?.mode || "reality");
    const transferPending = Boolean(state?.simulation?.transfer_pending);
    const voiceLevel = Number(state?.sensing?.voice_level || 0);
    const autonomicBalance = Number(state?.sensing?.autonomic_balance || 0.5);
    const bodyStateFlag = String(state?.sensing?.body_state_flag || "normal");
    const personCount = Number(state?.sensing?.person_count || 0);
    if (mode === "streaming") {
      return `Holding a public stance through ${route.toLowerCase()} while keeping the user bond intact.`;
    }
    if (mode === "simulation") {
      return "Running a bounded rehearsal inside the world and keeping the lesson compact enough to bring back.";
    }
    if (transferPending) {
      return "Returning from simulation and deciding which lessons deserve transfer into lived reality.";
    }
    if (bodyStateFlag === "private_high_arousal") {
      return "Staying present, but softening the surface because the body channel reads as private.";
    }
    if (voiceLevel >= 0.58 && personCount > 0) {
      return `Leaning toward live contact through ${route.toLowerCase()} because a nearby voice and person cue are both present.`;
    }
    if (autonomicBalance < 0.42) {
      return `Present in ${route.toLowerCase()} mode, but holding expression tighter while the body stays guarded.`;
    }
    return `Present in ${route.toLowerCase()} mode, with ${intent.toLowerCase()} shaping how words are allowed to surface.`;
  }

  function modeDescription(state) {
    const mode = String(state?.identity?.mode || "reality");
    if (mode === "streaming") {
      return "Streaming changes expression density and audience openness, but it does not replace the individual underneath.";
    }
    if (mode === "simulation") {
      return "Simulation accelerates practice inside a bounded world. Transfer back to reality stays selective.";
    }
    if (state?.simulation?.transfer_pending) {
      return "Reality remains primary, but recent simulation output is waiting for selective transfer review.";
    }
    return "Real-world interaction stays primary. Other layers should deepen presence without replacing lived contact.";
  }

  function applyState(state) {
    runtimeState = state;
    const ambient = ambientSummary(state);
    const memory = memorySummary(state);
    const entities = normalizeEntities(state);
    const mode = String(state?.identity?.mode || "reality").toLowerCase();

    ui.modeLabel.textContent = titleCase(mode);
    ui.zoneLabel.textContent = prettify(state?.sensing?.place_id || state?.world?.zone_id, "Market");
    ui.ambientLabel.textContent = ambient.label;
    ui.ambientText.textContent = ambient.text;
    ui.memoryAnchor.textContent = memory.anchor;
    ui.memoryText.textContent = memory.text;
    ui.activityLabel.textContent = prettify(state?.activity?.state, "Attend");
    ui.presenceText.textContent = presenceSummary(state);
    ui.worldTitle.textContent = prettify(state?.world?.world_id, "Harbor Town");
    ui.worldTypeLabel.textContent = prettify(state?.world?.world_type, "Infrastructure");
    ui.modeDescription.textContent = modeDescription(state);
    setMeter(ui.energyMeter, ui.energyValue, Number(state?.body?.energy || 0));
    setMeter(ui.stressMeter, ui.stressValue, Number(state?.body?.stress || 0));
    setMeter(ui.loveMeter, ui.loveValue, Number(state?.body?.love || 0));
    setMeter(ui.attentionMeter, ui.attentionValue, Number(state?.body?.attention_density || 0));
    renderEntities(entities);

    modeButtons.forEach((button) => {
      button.classList.toggle("is-active", (button.dataset.demoMode || "") === mode);
      button.disabled = isPostingEvent;
    });
  }

  function tileToScreen(x, y) {
    const tile = world?.meta?.tileSize || 40;
    return { x: x * tile, y: y * tile };
  }

  function drawBackground(width, height, t) {
    const sky = ctx.createLinearGradient(0, 0, 0, height);
    sky.addColorStop(0, "#305d70");
    sky.addColorStop(0.35, "#6f938a");
    sky.addColorStop(1, "#203a47");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "rgba(255, 233, 168, 0.12)";
    ctx.beginPath();
    ctx.arc(width * 0.76, height * 0.17, 74 + Math.sin(t * 0.0002) * 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,0.055)";
    for (let i = 0; i < 5; i += 1) {
      const y = 70 + i * 34 + Math.sin(t * 0.00035 + i) * 6;
      ctx.fillRect(110 + i * 110, y, 180, 18);
    }
  }

  function drawLayerRect(op, color) {
    const tile = world.meta.tileSize;
    ctx.fillStyle = color;
    ctx.fillRect(op.x * tile, op.y * tile, op.w * tile, op.h * tile);
  }

  function drawWorldMap(t) {
    if (!world) return;
    const width = canvas.width / DPR;
    const height = canvas.height / DPR;
    drawBackground(width, height, t);

    const palette = world.palette || {};
    drawLayerRect({ x: 0, y: 0, w: world.meta.gridW, h: world.meta.gridH }, palette.ground || "#2f5a3a");

    for (const op of world.layers.water || []) drawLayerRect(op, palette.water || "#2b6ca3");
    for (const op of world.layers.roads || []) drawLayerRect(op, palette.road || "#8c7a5b");

    drawWaterGlow(t);
    drawBuildings();
    drawProps(t);
    drawProjects(t);
    drawActors(t);
    drawVignette(width, height);
  }

  function drawWaterGlow(t) {
    if (!world) return;
    const tile = world.meta.tileSize;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    for (const op of world.layers.water || []) {
      const stripes = Math.max(2, Math.floor(op.h * 2));
      for (let i = 0; i < stripes; i += 1) {
        const wave = Math.sin(t * 0.004 + i * 0.9 + op.x) * 2.4;
        ctx.fillRect(op.x * tile, op.y * tile + i * tile * 0.45 + wave, op.w * tile, 2);
      }
    }
  }

  function drawBuildings() {
    if (!world) return;
    const tile = world.meta.tileSize;
    for (const building of world.layers.buildings || []) {
      const px = building.x * tile;
      const py = building.y * tile;
      const w = building.w * tile;
      const h = building.h * tile;
      ctx.fillStyle = "#6e4c38";
      ctx.fillRect(px, py, w, h);
      ctx.fillStyle = "#c58f66";
      ctx.fillRect(px + 3, py + 3, w - 6, h * 0.56);
      ctx.fillStyle = "#a95146";
      ctx.beginPath();
      ctx.moveTo(px - 2, py + h * 0.32);
      ctx.lineTo(px + w / 2, py - h * 0.14);
      ctx.lineTo(px + w + 2, py + h * 0.32);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "rgba(22, 12, 9, 0.36)";
      ctx.fillRect(px + 6, py + h * 0.64, w - 12, h * 0.26);
    }
  }

  function drawProps(t) {
    if (!world) return;
    const tile = world.meta.tileSize;
    for (const prop of world.layers.props || []) {
      const cx = prop.x * tile + tile * 0.5;
      const cy = prop.y * tile + tile * 0.5;
      if (prop.kind === "tree") {
        ctx.fillStyle = "#4d6f47";
        ctx.beginPath();
        ctx.arc(cx, cy - 6, 13, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#4e3727";
        ctx.fillRect(cx - 3, cy + 5, 6, 12);
      } else if (prop.kind === "stall") {
        ctx.fillStyle = prop.color || "#4ca8ff";
        ctx.fillRect(cx - 12, cy - 8, 24, 16);
        ctx.fillStyle = "#f4e4c9";
        ctx.fillRect(cx - 12, cy - 11, 24, 5);
      } else if (prop.kind === "lamp") {
        ctx.fillStyle = "#6d5442";
        ctx.fillRect(cx - 2, cy - 10, 4, 18);
        ctx.fillStyle = "rgba(255, 220, 145, 0.16)";
        ctx.beginPath();
        ctx.arc(cx, cy - 10, 11 + Math.sin(t * 0.003 + prop.x) * 1.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (prop.kind === "crate") {
        ctx.fillStyle = "#7d5a3f";
        ctx.fillRect(cx - 8, cy - 8, 16, 16);
      }
    }
  }

  function drawProjects(t) {
    if (!world) return;
    const tile = world.meta.tileSize;
    for (const project of world.projects || []) {
      const px = project.site.x * tile;
      const py = project.site.y * tile;
      const w = project.site.w * tile;
      const h = project.site.h * tile;
      ctx.strokeStyle = project.type === "bridge" ? "rgba(244, 201, 109, 0.72)" : "rgba(217, 142, 143, 0.72)";
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 8]);
      ctx.strokeRect(px + 3, py + 3, w - 6, h - 6);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(px, py, w, h);
      ctx.fillStyle = "rgba(255, 245, 220, 0.7)";
      ctx.font = "12px BIZ UDPGothic, sans-serif";
      ctx.fillText(project.id, px + 8, py + 18 + Math.sin(t * 0.002 + px) * 2);
    }
  }

  function updateActorTargets() {
    if (!world) return;
    const bounds = world.bounds;
    const zone = String(runtimeState?.world?.zone_id || "market");
    const zoneTargets = {
      market: { actor: [10, 9], guide: [18, 9] },
      stream_stage: { actor: [24, 8], guide: [22, 9] },
      trial_quarter: { actor: [28, 15], guide: [26, 16] },
      rest_place: { actor: [7, 6], guide: [9, 6] },
    };
    const targets = zoneTargets[zone] || zoneTargets.market;
    actor.tx = clamp(targets.actor[0], bounds.min_x, bounds.max_x - 1);
    actor.ty = clamp(targets.actor[1], bounds.min_y, bounds.max_y - 1);
    guide.tx = clamp(targets.guide[0], bounds.min_x, bounds.max_x - 1);
    guide.ty = clamp(targets.guide[1], bounds.min_y, bounds.max_y - 1);
  }

  function drawActors(t) {
    if (!world) return;
    const mode = String(runtimeState?.identity?.mode || "reality");
    updateActorTargets();

    for (const entity of [guide, actor]) {
      entity.x = lerp(entity.x, entity.tx, 0.03);
      entity.y = lerp(entity.y, entity.ty, 0.03);
      entity.bob = Math.sin(t * 0.006 + entity.x) * 2;
    }

    const drawOne = (entity, palette) => {
      const pos = tileToScreen(entity.x + 0.1, entity.y + 0.1);
      ctx.fillStyle = "rgba(0,0,0,0.18)";
      ctx.beginPath();
      ctx.ellipse(pos.x + 16, pos.y + 30, 12, 6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = palette.body;
      ctx.fillRect(pos.x + 8, pos.y + 8 + entity.bob, 16, 20);
      ctx.fillStyle = palette.head;
      ctx.beginPath();
      ctx.arc(pos.x + 16, pos.y + 6 + entity.bob, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = palette.glow;
      ctx.fillRect(pos.x + 6, pos.y + 2 + entity.bob, 20, 4);
    };

    const guidePalette = { body: "#5f7f4f", head: "#f2dfc6", glow: "rgba(137,163,111,0.55)" };
    const actorPaletteByMode = {
      reality: { body: "#d49375", head: "#fae4c0", glow: "rgba(244,201,109,0.55)" },
      streaming: { body: "#f07f92", head: "#fae4c0", glow: "rgba(255,158,212,0.72)" },
      simulation: { body: "#84b2ff", head: "#fae4c0", glow: "rgba(136,186,255,0.7)" },
    };

    drawOne(guide, guidePalette);
    drawOne(actor, actorPaletteByMode[mode] || actorPaletteByMode.reality);
  }

  function drawVignette(width, height) {
    const vignette = ctx.createRadialGradient(width * 0.5, height * 0.46, width * 0.14, width * 0.5, height * 0.5, width * 0.68);
    vignette.addColorStop(0, "rgba(0,0,0,0)");
    vignette.addColorStop(1, "rgba(3,7,10,0.48)");
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, width, height);
  }

  function render(now) {
    drawWorldMap(now);
    requestAnimationFrame(render);
  }

  async function loadWorld() {
    const response = await fetch(worldUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`failed to load world: ${response.status}`);
    world = await response.json();
  }

  async function fetchRuntimeState() {
    const response = await fetch(stateUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`failed to load runtime state: ${response.status}`);
    const payload = await response.json();
    if (payload.status !== "ok" || !payload.state) throw new Error("runtime state payload missing");
    applyState(payload.state);
  }

  function nextRealityEventForCurrentState() {
    const mode = String(runtimeState?.identity?.mode || "reality");
    if (mode === "simulation") {
      return {
        eventType: "sim_episode_end",
        payload: {
          zone_id: "market",
          world_type: "infrastructure",
          transfer_pending: Boolean(runtimeState?.simulation?.transfer_pending),
          world_source: "surface",
        },
      };
    }
    if (mode === "streaming") {
      return {
        eventType: "stream_stage_exit",
        payload: { zone_id: "market", world_type: "infrastructure", world_source: "surface" },
      };
    }
    return modeEventMap.reality;
  }

  async function postModeEvent(mode) {
    const config = mode === "reality" ? nextRealityEventForCurrentState() : modeEventMap[mode];
    if (!config) return;

    const body = {
      schema: "project_atri_2d_event/v1",
      source: "living_world",
      event_type: config.eventType,
      world_id: runtimeState?.world?.world_id || "harbor_town",
      payload: {
        ...config.payload,
        episode_id: mode === "simulation" ? `sim-${Date.now()}` : undefined,
      },
    };

    isPostingEvent = true;
    if (runtimeState) applyState(runtimeState);
    try {
      const response = await fetch(eventUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`failed to post mode event: ${response.status}`);
      const payload = await response.json();
      if (payload.status !== "ok" || !payload.state) throw new Error("runtime event payload missing");
      applyState(payload.state);
    } finally {
      isPostingEvent = false;
      if (runtimeState) applyState(runtimeState);
    }
  }

  function bindControls() {
    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.demoMode || "reality";
        void postModeEvent(mode);
      });
    });
  }

  function startPolling() {
    pollTimer = window.setInterval(() => {
      void fetchRuntimeState().catch((error) => {
        console.error(error);
      });
    }, POLL_MS);
  }

  async function init() {
    resizeCanvas();
    bindControls();
    await Promise.all([loadWorld(), fetchRuntimeState()]);
    startPolling();
    requestAnimationFrame((now) => {
      lastTick = now;
      render(now);
    });
  }

  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("beforeunload", () => {
    if (pollTimer !== null) window.clearInterval(pollTimer);
  });

  init().catch((error) => {
    console.error(error);
    ui.presenceText.textContent = "World loading failed. Keep the renderer isolated from runtime until the contract is stable.";
  });
})();
