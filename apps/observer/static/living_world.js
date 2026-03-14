(() => {
  const shell = document.querySelector('.living-shell');
  const canvas = document.getElementById('worldCanvas');
  const ctx = canvas.getContext('2d');
  const worldUrl = shell?.dataset.worldUrl || '/worlds/harbor_town.json';
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

  const ui = {
    modeLabel: document.getElementById('modeLabel'),
    zoneLabel: document.getElementById('zoneLabel'),
    ambientLabel: document.getElementById('ambientLabel'),
    ambientText: document.getElementById('ambientText'),
    memoryAnchor: document.getElementById('memoryAnchor'),
    memoryText: document.getElementById('memoryText'),
    activityLabel: document.getElementById('activityLabel'),
    presenceText: document.getElementById('presenceText'),
    worldTitle: document.getElementById('worldTitle'),
    worldTypeLabel: document.getElementById('worldTypeLabel'),
    entityList: document.getElementById('entityList'),
    modeDescription: document.getElementById('modeDescription'),
    energyMeter: document.getElementById('energyMeter'),
    stressMeter: document.getElementById('stressMeter'),
    loveMeter: document.getElementById('loveMeter'),
    attentionMeter: document.getElementById('attentionMeter'),
    energyValue: document.getElementById('energyValue'),
    stressValue: document.getElementById('stressValue'),
    loveValue: document.getElementById('loveValue'),
    attentionValue: document.getElementById('attentionValue'),
  };

  const modeButtons = [...document.querySelectorAll('.mode-chip')];
  const stateTimeline = {
    reality: [
      {
        mode: 'Reality',
        zone: 'Harbor Market',
        worldType: 'infrastructure',
        worldTitle: 'Harbor Town',
        ambientLabel: 'settled daylight',
        ambientText: 'The companion stays near the user and lets the square breathe.',
        activity: 'attend',
        presence: 'Listening closely and letting the place settle before speaking.',
        anchor: 'bakery',
        memoryText: 'A place-linked recall is warm but not overwhelming.',
        values: { energy: 0.72, stress: 0.24, love: 0.44, attention: 0.57 },
        modeDescription: 'Real-world interaction stays primary. Other modes only change expression density.',
        entities: [
          { name: 'user', kind: 'bonded presence' },
          { name: 'guide', kind: 'nearby ally' },
          { name: 'vendor_01', kind: 'ambient NPC' },
        ],
      },
      {
        mode: 'Reality',
        zone: 'Memory Spot',
        worldType: 'infrastructure',
        worldTitle: 'Harbor Town',
        ambientLabel: 'quiet ignition',
        ambientText: 'A familiar facade catches the eye and opens a short recall chain.',
        activity: 'recall',
        presence: 'Noticing a remembered shape before turning it into words.',
        anchor: 'market corner',
        memoryText: 'Visual cues are gently re-lighting an older route memory.',
        values: { energy: 0.68, stress: 0.22, love: 0.5, attention: 0.66 },
        modeDescription: 'Memory should color the scene, not swallow it.',
        entities: [
          { name: 'user', kind: 'bonded presence' },
          { name: 'stall_blue', kind: 'cue object' },
          { name: 'dock_worker_01', kind: 'ambient NPC' },
        ],
      },
    ],
    streaming: [
      {
        mode: 'Streaming',
        zone: 'Stage Edge',
        worldType: 'infrastructure',
        worldTitle: 'Harbor Stage',
        ambientLabel: 'public shimmer',
        ambientText: 'Expression brightens while the same individual remains underneath.',
        activity: 'stream',
        presence: 'Opening toward many people without becoming a different person.',
        anchor: 'stage light',
        memoryText: 'Shared moments are being collected as atmosphere, not just metrics.',
        values: { energy: 0.77, stress: 0.35, love: 0.58, attention: 0.74 },
        modeDescription: 'Streaming raises expression amplitude, not identity replacement.',
        entities: [
          { name: 'audience_glow', kind: 'social field' },
          { name: 'user', kind: 'bond anchor' },
          { name: 'guide', kind: 'support presence' },
        ],
      },
    ],
    simulation: [
      {
        mode: 'Simulation',
        zone: 'Trial Quarter',
        worldType: 'community',
        worldTitle: 'Mini World',
        ambientLabel: 'accelerated rehearsal',
        ambientText: 'The world speeds up practice but does not replace lived reality.',
        activity: 'simulate',
        presence: 'Testing recovery and coordination inside a bounded world.',
        anchor: 'bridge trial',
        memoryText: 'Only compact lessons should return to reality from here.',
        values: { energy: 0.64, stress: 0.29, love: 0.41, attention: 0.69 },
        modeDescription: 'Simulation accelerates growth, but transfer back must stay selective.',
        entities: [
          { name: 'worker_route', kind: 'sim actor' },
          { name: 'bridge_01', kind: 'world project' },
          { name: 'user_shadow', kind: 'training anchor' },
        ],
      },
    ],
  };

  let world = null;
  let activeMode = 'reality';
  let stateIndex = 0;
  let lastTick = performance.now();
  let cycleElapsed = 0;
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

  function currentState() {
    const timeline = stateTimeline[activeMode] || stateTimeline.reality;
    return timeline[stateIndex % timeline.length];
  }

  function setMode(mode) {
    activeMode = mode;
    stateIndex = 0;
    modeButtons.forEach((button) => {
      button.classList.toggle('is-active', button.dataset.demoMode === mode);
    });
    applyState(currentState());
  }

  function applyState(state) {
    ui.modeLabel.textContent = state.mode;
    ui.zoneLabel.textContent = state.zone;
    ui.ambientLabel.textContent = state.ambientLabel;
    ui.ambientText.textContent = state.ambientText;
    ui.memoryAnchor.textContent = state.anchor;
    ui.memoryText.textContent = state.memoryText;
    ui.activityLabel.textContent = state.activity;
    ui.presenceText.textContent = state.presence;
    ui.worldTitle.textContent = state.worldTitle;
    ui.worldTypeLabel.textContent = state.worldType;
    ui.modeDescription.textContent = state.modeDescription;
    setMeter(ui.energyMeter, ui.energyValue, state.values.energy);
    setMeter(ui.stressMeter, ui.stressValue, state.values.stress);
    setMeter(ui.loveMeter, ui.loveValue, state.values.love);
    setMeter(ui.attentionMeter, ui.attentionValue, state.values.attention);
    renderEntities(state.entities);
  }

  function setMeter(meter, label, value) {
    const pct = `${Math.round(clamp(value, 0, 1) * 100)}%`;
    meter.style.width = pct;
    label.textContent = value.toFixed(2);
  }

  function renderEntities(entities) {
    ui.entityList.innerHTML = '';
    for (const entity of entities) {
      const item = document.createElement('li');
      item.innerHTML = `<div><strong>${entity.name}</strong><div class="entity-kind">${entity.kind}</div></div><span class="scene-badge">active</span>`;
      ui.entityList.appendChild(item);
    }
  }

  function tileToScreen(x, y) {
    const tile = world?.meta?.tileSize || 40;
    return { x: x * tile, y: y * tile };
  }

  function drawBackground(width, height, t) {
    const sky = ctx.createLinearGradient(0, 0, 0, height);
    sky.addColorStop(0, '#305d70');
    sky.addColorStop(0.35, '#6f938a');
    sky.addColorStop(1, '#203a47');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = 'rgba(255, 233, 168, 0.12)';
    ctx.beginPath();
    ctx.arc(width * 0.76, height * 0.17, 74 + Math.sin(t * 0.0002) * 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(255,255,255,0.055)';
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
    drawLayerRect({ x: 0, y: 0, w: world.meta.gridW, h: world.meta.gridH }, palette.ground || '#2f5a3a');

    for (const op of world.layers.water || []) {
      drawLayerRect(op, palette.water || '#2b6ca3');
    }
    for (const op of world.layers.roads || []) {
      drawLayerRect(op, palette.road || '#8c7a5b');
    }

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
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
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
      ctx.fillStyle = '#6e4c38';
      ctx.fillRect(px, py, w, h);
      ctx.fillStyle = '#c58f66';
      ctx.fillRect(px + 3, py + 3, w - 6, h * 0.56);
      ctx.fillStyle = '#a95146';
      ctx.beginPath();
      ctx.moveTo(px - 2, py + h * 0.32);
      ctx.lineTo(px + w / 2, py - h * 0.14);
      ctx.lineTo(px + w + 2, py + h * 0.32);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = 'rgba(22, 12, 9, 0.36)';
      ctx.fillRect(px + 6, py + h * 0.64, w - 12, h * 0.26);
    }
  }

  function drawProps(t) {
    if (!world) return;
    const tile = world.meta.tileSize;
    for (const prop of world.layers.props || []) {
      const cx = prop.x * tile + tile * 0.5;
      const cy = prop.y * tile + tile * 0.5;
      if (prop.kind === 'tree') {
        ctx.fillStyle = '#4d6f47';
        ctx.beginPath();
        ctx.arc(cx, cy - 6, 13, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#4e3727';
        ctx.fillRect(cx - 3, cy + 5, 6, 12);
      } else if (prop.kind === 'stall') {
        ctx.fillStyle = prop.color || '#4ca8ff';
        ctx.fillRect(cx - 12, cy - 8, 24, 16);
        ctx.fillStyle = '#f4e4c9';
        ctx.fillRect(cx - 12, cy - 11, 24, 5);
      } else if (prop.kind === 'lamp') {
        ctx.fillStyle = '#6d5442';
        ctx.fillRect(cx - 2, cy - 10, 4, 18);
        ctx.fillStyle = 'rgba(255, 220, 145, 0.16)';
        ctx.beginPath();
        ctx.arc(cx, cy - 10, 11 + Math.sin(t * 0.003 + prop.x) * 1.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (prop.kind === 'crate') {
        ctx.fillStyle = '#7d5a3f';
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
      ctx.strokeStyle = project.type === 'bridge' ? 'rgba(244, 201, 109, 0.72)' : 'rgba(217, 142, 143, 0.72)';
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 8]);
      ctx.strokeRect(px + 3, py + 3, w - 6, h - 6);
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.05)';
      ctx.fillRect(px, py, w, h);
      ctx.fillStyle = 'rgba(255, 245, 220, 0.7)';
      ctx.font = '12px BIZ UDPGothic, sans-serif';
      ctx.fillText(project.id, px + 8, py + 18 + Math.sin(t * 0.002 + px) * 2);
    }
  }

  function drawActors(t) {
    if (!world) return;
    const tile = world.meta.tileSize;
    updateActorTargets(t);

    for (const entity of [guide, actor]) {
      entity.x = lerp(entity.x, entity.tx, 0.03);
      entity.y = lerp(entity.y, entity.ty, 0.03);
      entity.bob = Math.sin(t * 0.006 + entity.x) * 2;
    }

    const drawOne = (entity, palette) => {
      const pos = tileToScreen(entity.x + 0.1, entity.y + 0.1);
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
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

    drawOne(guide, { body: '#5f7f4f', head: '#f2dfc6', glow: 'rgba(137,163,111,0.55)' });
    drawOne(actor, { body: '#d49375', head: '#fae4c0', glow: 'rgba(244,201,109,0.55)' });
  }

  function updateActorTargets(t) {
    if (!world) return;
    const bounds = world.bounds;
    const swing = Math.sin(t * 0.0004);
    actor.tx = clamp(10 + swing * 3.4, bounds.min_x, bounds.max_x - 1);
    actor.ty = clamp(9 + Math.cos(t * 0.0003) * 1.4, bounds.min_y, bounds.max_y - 1);
    guide.tx = clamp(18 + Math.cos(t * 0.0005) * 2.6, bounds.min_x, bounds.max_x - 1);
    guide.ty = clamp(9 + Math.sin(t * 0.00045) * 1.8, bounds.min_y, bounds.max_y - 1);
  }

  function drawVignette(width, height) {
    const vignette = ctx.createRadialGradient(width * 0.5, height * 0.46, width * 0.14, width * 0.5, height * 0.5, width * 0.68);
    vignette.addColorStop(0, 'rgba(0,0,0,0)');
    vignette.addColorStop(1, 'rgba(3,7,10,0.48)');
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, width, height);
  }

  function advanceState(dt) {
    cycleElapsed += dt;
    const holdMs = activeMode === 'reality' ? 6500 : 7200;
    const timeline = stateTimeline[activeMode] || stateTimeline.reality;
    if (timeline.length > 1 && cycleElapsed >= holdMs) {
      cycleElapsed = 0;
      stateIndex = (stateIndex + 1) % timeline.length;
      applyState(currentState());
    }
  }

  function render(now) {
    const dt = now - lastTick;
    lastTick = now;
    advanceState(dt);
    drawWorldMap(now);
    requestAnimationFrame(render);
  }

  async function loadWorld() {
    const response = await fetch(worldUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`failed to load world: ${response.status}`);
    world = await response.json();
  }

  function bindControls() {
    modeButtons.forEach((button) => {
      button.addEventListener('click', () => setMode(button.dataset.demoMode || 'reality'));
    });
  }

  async function init() {
    resizeCanvas();
    bindControls();
    applyState(currentState());
    await loadWorld();
    requestAnimationFrame((now) => {
      lastTick = now;
      render(now);
    });
  }

  window.addEventListener('resize', resizeCanvas);
  init().catch((error) => {
    console.error(error);
    ui.presenceText.textContent = 'World loading failed. Keep the renderer isolated from runtime until the contract is stable.';
  });
})();
