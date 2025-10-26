"""Minimal web dashboard for EQNet live metrics and events."""

from __future__ import annotations

import argparse
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Optional


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>EQNet Live Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: "Segoe UI", "Hiragino Sans", sans-serif;
      background-color: #111;
      color: #f3f3f3;
    }}
    body {{
      margin: 0;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-width: 960px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}
    h1 {{
      font-size: 1.4rem;
      margin: 0;
    }}
    .status {{
      font-size: 0.9rem;
      opacity: 0.75;
    }}
    .card {{
      background: #1c1c1c;
      border: 1px solid #2d2d2d;
      border-radius: 12px;
      padding: 1rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }}
    .card h2 {{
      grid-column: 1 / -1;
      margin: 0 0 0.5rem 0;
      font-size: 1rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    canvas {{
      background: #101010;
      border-radius: 8px;
      border: 1px solid #242424;
      width: 100%;
      height: 120px;
    }}
    .metrics-values {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem;
      font-size: 0.95rem;
    }}
    .metric-label {{
      opacity: 0.7;
      font-size: 0.85rem;
    }}
    .controls-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
      font-size: 0.95rem;
    }}
    .badge {{
      display: inline-flex;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      align-items: center;
      gap: 0.35rem;
      letter-spacing: 0.02em;
    }}
    .badge.green {{
      background-color: rgba(46, 204, 113, 0.15);
      color: #2ecc71;
    }}
    .badge.amber {{
      background-color: rgba(241, 196, 15, 0.15);
      color: #f1c40f;
    }}
    .badge.red {{
      background-color: rgba(231, 76, 60, 0.15);
      color: #e74c3c;
    }}
    .badge.purple {{
      background-color: rgba(155, 89, 182, 0.18);
      color: #dcb0ff;
    }}
    .bud-card {{
      grid-template-columns: 1fr 1fr;
      align-items: stretch;
    }}
    .lights-card {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      text-align: center;
    }}
    .light-indicator {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.4rem;
    }}
    .light-circle {{
      width: 42px;
      height: 42px;
      border-radius: 50%;
      border: 2px solid #2d2d2d;
      background: #1a1a1a;
      transition: background 0.3s ease, box-shadow 0.3s ease;
    }}
    .light-circle.green {{
      background: radial-gradient(circle, rgba(56, 180, 102, 0.9) 0%, rgba(15, 70, 31, 0.4) 60%);
      box-shadow: 0 0 12px rgba(56, 180, 102, 0.7);
    }}
    .light-circle.blue {{
      background: radial-gradient(circle, rgba(80, 140, 220, 0.9) 0%, rgba(20, 50, 120, 0.4) 60%);
      box-shadow: 0 0 12px rgba(50, 110, 210, 0.6);
    }}
    .light-circle.red {{
      background: radial-gradient(circle, rgba(220, 80, 80, 0.9) 0%, rgba(120, 20, 20, 0.4) 60%);
      box-shadow: 0 0 12px rgba(220, 80, 80, 0.6);
    }}
    .light-label {{
      font-size: 0.85rem;
      letter-spacing: 0.03em;
      opacity: 0.8;
    }}
    .network-card {{
      grid-template-columns: 1fr;
      position: relative;
    }}
    #budCanvas {{
      width: 100%;
      height: 190px;
      border: 1px solid #2c1f38;
      border-radius: 10px;
      background: radial-gradient(circle at center, rgba(155, 89, 182, 0.12), #0b0710 70%);
    }}
    .bud-meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.6rem;
      font-size: 0.95rem;
    }}
    .bud-meta .metric-label {{
      opacity: 0.65;
      font-size: 0.82rem;
    }}
    footer {{
      font-size: 0.75rem;
      opacity: 0.6;
    }}
    #networkCanvas {{
      width: 100%;
      height: 200px;
      border: 1px solid #1f2e38;
      border-radius: 10px;
      background: #080c11;
    }}
    #rSpark {{
      width: 100%;
      height: 80px;
      border-radius: 6px;
      background: linear-gradient(180deg, rgba(120,10,10,0.2) 0%, rgba(10,10,10,0.0) 35%);
      border: 1px solid #322a36;
      margin-top: 0.8rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>EQNet Live Dashboard</h1>
    <div class="status" id="status">ws://{bus_host}:{bus_port}</div>
  </header>
  <section class="card">
    <h2>ﾎ｣ / ﾎｨ Sparkline</h2>
    <canvas id="sigmaSpark" width="420" height="120"></canvas>
    <canvas id="psiSpark" width="420" height="120"></canvas>
    <div class="metrics-values">
      <div>
        <div class="metric-label">ﾎ｣ (sigma)</div>
        <div id="sigmaValue">--</div>
      </div>
      <div>
        <div class="metric-label">ﾎｨ (psi)</div>
        <div id="psiValue">--</div>
      </div>
      <div>
        <div class="metric-label">CAM ﾎ｣ / ﾎｨ</div>
        <div id="sigmaCam">--</div>
      </div>
      <div>
        <div class="metric-label">AUDIO ﾎ｣ / ﾎｨ</div>
        <div id="sigmaAudio">--</div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Controls Snapshot</h2>
    <div class="controls-grid">
      <div>
        <div class="metric-label">warmth</div>
        <div id="warmthValue">--</div>
      </div>
      <div>
        <div class="metric-label">pause_ms</div>
        <div id="pauseValue">--</div>
      </div>
      <div>
        <div class="metric-label">motion_speed</div>
        <div id="motionValue">--</div>
      </div>
      <div>
        <div class="metric-label">containment</div>
        <div id="containmentBadge" class="badge amber">unknown</div>
      </div>
      <div>
        <div class="metric-label">health</div>
        <div id="healthBadge" class="badge green">ok</div>
      </div>
      <div>
        <div class="metric-label">privacy</div>
        <div id="privacyBadge" class="badge green">camera_on_no_storage</div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Last Event</h2>
    <div id="lastEvent">縺ｾ縺繧､繝吶Φ繝医・縺ゅｊ縺ｾ縺帙ｓ</div>
  </section>

  <section class="card bud-card">
    <h2>闃ｽ蜷ｹ縺阪Δ繝ｼ繝・/h2>
    <canvas id="budCanvas" width="256" height="256"></canvas>
    <div class="bud-meta">
      <div>
        <div class="metric-label">score</div>
        <div id="budScore">--</div>
      </div>
      <div>
        <div class="metric-label">ﾏ・(spectral)</div>
        <div id="budRho">--</div>
      </div>
      <div>
        <div class="metric-label">coords</div>
        <div id="budCoords">--</div>
      </div>
      <div>
        <div class="metric-label">novelty/meta</div>
        <div id="budFeatures">--</div>
      </div>
      <div>
        <div class="metric-label">迥ｶ諷・/div>
        <div id="budBadge" class="badge purple">inactive</div>
      </div>
    </div>
  </section>

  <section class="card lights-card">
    <div class="light-indicator">
      <div class="light-circle" id="lightBud"></div>
      <div class="light-label">Bud</div>
    </div>
    <div class="light-indicator">
      <div class="light-circle" id="lightR"></div>
      <div class="light-label">Synchrony R</div>
    </div>
    <div class="light-indicator">
      <div class="light-circle" id="lightRho"></div>
      <div class="light-label">Field ﾏ・/div>
    </div>
  </section>

  <section class="card network-card">
    <h2>繝阪ャ繝医Ρ繝ｼ繧ｯ蜷梧悄</h2>
    <canvas id="networkCanvas" width="360" height="200"></canvas>
    <canvas id="rSpark" width="360" height="80"></canvas>
  </section>

  <footer>
    Hotkeys: F9 繧ｻ繝・す繝ｧ繝ｳ / F10 髻ｳ莉句・荳譎ょ●豁｢ / F11 繝ｭ繧ｰ繝槭・繧ｫ繝ｼ 繝ｻ Ctrl+R 縺ｧ config reload
  </footer>

  <script>
    const BUS_WS = "ws://{bus_host}:{bus_port}";
    const sigmaHistory = [];
    const psiHistory = [];
    const MAX_HISTORY = 180;

    const sigmaCanvas = document.getElementById("sigmaSpark");
    const psiCanvas = document.getElementById("psiSpark");
    const budCanvas = document.getElementById("budCanvas");
    const budCtx = budCanvas.getContext("2d");
    const netCanvas = document.getElementById("networkCanvas");
    const netCtx = netCanvas.getContext("2d");
    const rSpark = document.getElementById("rSpark");
    const rCtx = rSpark.getContext("2d");
    const budScoreEl = document.getElementById("budScore");
    const budRhoEl = document.getElementById("budRho");
    const budCoordsEl = document.getElementById("budCoords");
    const budFeatEl = document.getElementById("budFeatures");
    const budBadge = document.getElementById("budBadge");
    let budTimeout = null;
    const rHistory = [];
    const MAX_R_HISTORY = 200;
    const edgeFadeMs = 1200;
    let netArrows = [];
    const nodePositions = new Map();
    const nodeRadius = 16;
    const NETWORK_CENTER = {{ x: netCanvas.width / 2, y: netCanvas.height / 2 }};
    const NETWORK_R = Math.min(netCanvas.width, netCanvas.height) * 0.35;
    const lightBud = document.getElementById("lightBud");
    const lightR = document.getElementById("lightR");
    const lightRho = document.getElementById("lightRho");
    let budLightTimer = null;
    let lastRho = 0.0;

    function drawSparkline(canvas, series, color) {{
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (series.length < 2) {{
        return;
      }}
      const min = Math.min(...series);
      const max = Math.max(...series);
      const span = (max - min) || 1;
      ctx.beginPath();
      series.forEach((value, idx) => {{
        const x = (idx / (series.length - 1)) * canvas.width;
        const y = canvas.height - ((value - min) / span) * canvas.height;
        if (idx === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          ctx.lineTo(x, y);
        }}
      }});
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    function formatNumber(value) {{
      if (value === undefined || value === null) {{
        return "--";
      }}
      if (typeof value === "number") {{
        return value.toFixed(3);
      }}
      return String(value);
    }}

    function setBadge(elementId, value, type) {{
      const el = document.getElementById(elementId);
      if (!el) return;
      el.textContent = value;
      el.classList.remove("green", "amber", "red");
      if (type === "containment") {{
        el.classList.add(value ? "red" : "green");
      }} else if (type === "health") {{
        if (value === "ok") {{
          el.classList.add("green");
        }} else if (value === "degraded") {{
          el.classList.add("amber");
        }} else {{
          el.classList.add("red");
        }}
      }} else {{
        el.classList.add("green");
      }}
    }}

    function updateMetrics(payload) {{
      document.getElementById("sigmaValue").textContent = formatNumber(payload.sigma);
      document.getElementById("psiValue").textContent = formatNumber(payload.psi);
      document.getElementById("sigmaCam").textContent = `${{formatNumber(payload.sigma_cam)}} / ${{formatNumber(payload.psi_cam)}}`;
      document.getElementById("sigmaAudio").textContent = `${{formatNumber(payload.sigma_audio)}} / ${{formatNumber(payload.psi_audio)}}`;
      if (typeof payload.sigma === "number") {{
        sigmaHistory.push(payload.sigma);
        if (sigmaHistory.length > MAX_HISTORY) sigmaHistory.shift();
        drawSparkline(sigmaCanvas, sigmaHistory, "#3498db");
      }}
      if (typeof payload.psi === "number") {{
        psiHistory.push(payload.psi);
        if (psiHistory.length > MAX_HISTORY) psiHistory.shift();
        drawSparkline(psiCanvas, psiHistory, "#9b59b6");
      }}
      if (typeof payload.R === "number") {{
        rHistory.push(payload.R);
        if (rHistory.length > MAX_R_HISTORY) rHistory.shift();
        drawRSpark();
        updateLightsForR(payload.R);
      }}
    }}

    function updateControls(payload) {{
      document.getElementById("warmthValue").textContent = formatNumber(payload.warmth);
      document.getElementById("pauseValue").textContent = formatNumber(payload.pause_ms);
      document.getElementById("motionValue").textContent = formatNumber(payload.motion_speed);
      setBadge("containmentBadge", payload.containment ? "true" : "false", "containment");
      setBadge("healthBadge", payload.health || "ok", "health");
      setBadge("privacyBadge", payload.privacy_badge || "camera_on_no_storage", "privacy");
    }}

    function updateEvent(payload) {{
      const ts = payload.ts ? new Date(payload.ts).toLocaleTimeString() : "";
      const label = payload.name || payload.type || "event";
      document.getElementById("lastEvent").textContent = `${{ts}} :: ${{label}} [${{payload.level || "info"}}] ${{payload.notes || ""}}`;
    }}

    function clearBudGlow() {{
      budCtx.fillStyle = "#09060f";
      budCtx.fillRect(0, 0, budCanvas.width, budCanvas.height);
    }}

    function drawBudGlow(coords, gridSize = 64) {{
      clearBudGlow();
      if (!Array.isArray(coords) || coords.length === 0) {{
        return;
      }}
      const width = budCanvas.width;
      const height = budCanvas.height;
      coords.forEach(([x, y], idx) => {{
        const cx = ((x + 0.5) / gridSize) * width;
        const cy = ((y + 0.5) / gridSize) * height;
        const radius = 18 + idx * 6;
        const gradient = budCtx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        gradient.addColorStop(0, "rgba(223, 160, 255, 0.65)");
        gradient.addColorStop(0.5, "rgba(173, 109, 255, 0.25)");
        gradient.addColorStop(1, "rgba(15, 9, 25, 0.0)");
        budCtx.beginPath();
        budCtx.fillStyle = gradient;
        budCtx.arc(cx, cy, radius, 0, Math.PI * 2);
        budCtx.fill();
      }});
    }}

    function setBudInactive() {{
      budBadge.textContent = "inactive";
      budBadge.classList.remove("green", "amber");
      budBadge.classList.add("purple");
      budScoreEl.textContent = "--";
      budRhoEl.textContent = "--";
      budCoordsEl.textContent = "--";
      budFeatEl.textContent = "--";
      clearBudGlow();
    }}

    function setLight(element, state) {{
      if (!element) return;
      element.classList.remove("green", "blue", "red");
      if (state) {{
        element.classList.add(state);
      }}
    }}

    function updateLightsForR(Rvalue) {{
      if (!lightR) return;
      if (typeof Rvalue !== "number" || Number.isNaN(Rvalue)) {{
        setLight(lightR, null);
        return;
      }}
      setLight(lightR, Rvalue > 0.78 ? "red" : "blue");
    }}

    function updateLightsForRho(rhoValue) {{
      lastRho = rhoValue;
      if (!lightRho) return;
      if (typeof rhoValue !== "number" || Number.isNaN(rhoValue)) {{
        setLight(lightRho, null);
        return;
      }}
      setLight(lightRho, rhoValue > 1.8 ? "red" : "blue");
    }}

    function updateBudBadge(payload) {{
      budScoreEl.textContent = formatNumber(payload.score);
      budRhoEl.textContent = formatNumber(payload.rho);
      if (Array.isArray(payload.coords)) {{
        budCoordsEl.textContent = payload.coords.join(", ");
      }} else {{
        budCoordsEl.textContent = "--";
      }}
      budFeatEl.textContent = `${{formatNumber(payload.novelty)}} / ${{formatNumber(payload.meta)}}`;
      budBadge.classList.remove("green", "amber", "purple");
      if (payload.ok) {{
        budBadge.textContent = "active / safe";
        budBadge.classList.add("green");
      }} else {{
        budBadge.textContent = "active / pruning";
        budBadge.classList.add("amber");
      }}
      if (typeof payload.rho === "number") {{
        updateLightsForRho(payload.rho);
      }}
    }}

    function handleBudEvent(payload) {{
      updateBudBadge(payload);
      drawBudGlow(payload.bud_coords || []);
      if (budTimeout) {{
        clearTimeout(budTimeout);
      }}
      budTimeout = setTimeout(setBudInactive, 9000);
      if (lightBud) {{
        setLight(lightBud, "green");
        if (budLightTimer) {{
          clearTimeout(budLightTimer);
        }}
        budLightTimer = setTimeout(() => setLight(lightBud, null), 6000);
      }}
    }}

    function drawRSpark() {{
      rCtx.clearRect(0, 0, rSpark.width, rSpark.height);
      if (rHistory.length < 2) {{
        return;
      }}
      const maxVal = 1.0;
      rCtx.beginPath();
      rHistory.forEach((value, idx) => {{
        const x = (idx / (rHistory.length - 1)) * rSpark.width;
        const y = rSpark.height - (value / maxVal) * rSpark.height;
        if (idx === 0) {{
          rCtx.moveTo(x, y);
        }} else {{
          rCtx.lineTo(x, y);
        }}
      }});
      rCtx.strokeStyle = "#e67e22";
      rCtx.lineWidth = 2;
      rCtx.stroke();
    }}

    function updateNodePositions(nodes) {{
      const count = Math.max(1, nodes.length);
      nodes.forEach((name, idx) => {{
        const angle = (Math.PI * 2 * idx) / count;
        const x = NETWORK_CENTER.x + NETWORK_R * Math.cos(angle);
        const y = NETWORK_CENTER.y + NETWORK_R * Math.sin(angle);
        nodePositions.set(name, {{ x, y }});
      }});
    }}

    function drawNetwork(now = performance.now()) {{
      netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);
      const nodes = Array.from(nodePositions.keys()).sort();
      if (nodes.length === 0) {{
        updateNodePositions(["A", "B"]);
      }}
      netCtx.fillStyle = "#111820";
      netCtx.beginPath();
      netCtx.arc(NETWORK_CENTER.x, NETWORK_CENTER.y, NETWORK_R + 18, 0, Math.PI * 2);
      netCtx.fill();

      const active = [];
      netArrows.forEach((arrow) => {{
        const elapsed = now - arrow.created;
        if (elapsed < edgeFadeMs) {{
          const alpha = 1 - elapsed / edgeFadeMs;
          netCtx.strokeStyle = `rgba(231, 126, 35, ${{alpha.toFixed(2)}})`;
          netCtx.lineWidth = 2;
          netCtx.beginPath();
          netCtx.moveTo(arrow.from.x, arrow.from.y);
          netCtx.lineTo(arrow.to.x, arrow.to.y);
          netCtx.stroke();
          active.push(arrow);
        }}
      }});
      netArrows = active;

      Array.from(nodePositions.entries()).forEach(([name, pos]) => {{
        netCtx.fillStyle = "#1f2c3a";
        netCtx.beginPath();
        netCtx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
        netCtx.fill();
        netCtx.fillStyle = "#dce6f5";
        netCtx.font = "12px 'Segoe UI'";
        netCtx.textAlign = "center";
        netCtx.textBaseline = "middle";
        netCtx.fillText(name, pos.x, pos.y);
      }});
      requestAnimationFrame(drawNetwork);
    }}

    function handleNetEvent(payload) {{
      const actor = payload.from || payload.agent;
      const target = payload.to;
      const nodes = new Set(nodePositions.keys());
      if (actor) nodes.add(actor);
      if (target) nodes.add(target);
      if (nodes.size > 0) {{
        updateNodePositions(Array.from(nodes));
      }}
      if (payload.R !== undefined) {{
        rHistory.push(payload.R);
        if (rHistory.length > MAX_R_HISTORY) rHistory.shift();
        drawRSpark();
        updateLightsForR(payload.R);
      }}
      if (actor && target) {{
        const fromPos = nodePositions.get(actor);
        const toPos = nodePositions.get(target);
        if (fromPos && toPos) {{
          netArrows.push({{ from: fromPos, to: toPos, created: performance.now() }});
        }}
      }}
    }}

    function connectStream(path, handler) {{
      let socket;
      const connect = () => {{
        socket = new WebSocket(`${{BUS_WS}}/${{path}}`);
        socket.onopen = () => {{
          document.getElementById("status").textContent = `${{path}} connected`;
        }};
        socket.onmessage = (event) => {{
          try {{
            const payload = JSON.parse(event.data);
            handler(payload);
          }} catch (err) {{
            console.error("Failed to parse payload", err);
          }}
        }};
        socket.onclose = () => {{
          document.getElementById("status").textContent = `${{path}} reconnecting...`;
          setTimeout(connect, 1500);
        }};
        socket.onerror = () => {{
          document.getElementById("status").textContent = `${{path}} error`;
        }};
      }};
      connect();
    }}

    connectStream("metrics", updateMetrics);
    connectStream("controls", updateControls);
    connectStream("events", (payload) => {{
      if (payload.type === "bud") {{
        handleBudEvent(payload);
      }}
      if (payload.type === "net") {{
        handleNetEvent(payload);
      }}
      updateEvent(payload);
    }});
    setBudInactive();
    setLight(lightBud, null);
    setLight(lightR, null);
    setLight(lightRho, null);
    drawNetwork();
  </script>
</body>
</html>
"""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQNet live metrics and events")
    parser.add_argument("--listen-host", type=str, default="127.0.0.1", help="HTTP")
    parser.add_argument("--listen-port", type=int, default=8080, help="HTTP port")
    parser.add_argument(
        "--bus-host", type=str, default="127.0.0.1", help="EQNet bus host"
    )
    parser.add_argument("--bus-port", type=int, default=8765, help="EQNet bus port")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args(argv)


def make_handler(bus_host: str, bus_port: int) -> type[BaseHTTPRequestHandler]:
    html = DASHBOARD_HTML.format(bus_host=bus_host, bus_port=bus_port).encode("utf-8")

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: D401, N802
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_error(404, "Not Found")

        def log_message(self, format: str, *args) -> None:  # noqa: A003, D401
            logging.getLogger("eqnet.dashboard.http").info(
                "%s - %s", self.client_address[0], format % args
            )

    return DashboardHandler


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    )
    handler_cls = make_handler(args.bus_host, args.bus_port)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler_cls)
    logging.getLogger("eqnet.dashboard").info(
        "Dashboard available at http://%s:%s (bus ws://%s:%s)",
        args.listen_host,
        args.listen_port,
        args.bus_host,
        args.bus_port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("eqnet.dashboard").info("Shutting down dashboard...")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
