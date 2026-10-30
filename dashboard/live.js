// Shared connection to server.py's live event feed (ws://127.0.0.1:8767 by default).
// Override with ?feed=ws://host:port in the page URL.
const Live = (() => {
  const params = new URLSearchParams(location.search);
  const url = params.get("feed") || "ws://127.0.0.1:8767";
  const handlers = {};
  const linkHandlers = [];
  let retryMs = 500;

  function on(type, fn) { (handlers[type] ||= []).push(fn); }
  function onLink(fn) { linkHandlers.push(fn); }
  function setLink(up) { linkHandlers.forEach((fn) => fn(up)); }

  function connect() {
    const ws = new WebSocket(url);
    ws.onopen = () => { retryMs = 500; setLink(true); };
    ws.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      (handlers[ev.type] || []).forEach((fn) => fn(ev));
      (handlers["*"] || []).forEach((fn) => fn(ev));
    };
    ws.onclose = () => {
      setLink(false);
      setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, 4000);
    };
  }

  function clock(ts) {
    const d = ts ? new Date(ts * 1000) : new Date();
    return d.toLocaleTimeString([], { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
  }

  function setPill(el, cls, text) { el.className = "pill " + cls; el.textContent = text; }

  const STAGES = {
    waiting:      ["bad",  "No device connected"],
    standby:      ["ok",   "Standby: listening for “Khus Khus”"],
    choosing:     ["warn", "Wake word heard: choose ASR model in the server terminal"],
    listening:    ["wake", "Awake: transcribing"],
    transcribing: ["warn", "Running Whisper…"],
  };

  window.addEventListener("load", () => setTimeout(connect, 0));
  return { on, onLink, clock, setPill, STAGES, url };
})();
