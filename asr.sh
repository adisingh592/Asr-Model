#!/usr/bin/env bash
# Control script for the KWS + ASR server, the live windows, and the laptop-mic stand-in.
#
#   ./asr.sh start            server + live windows + laptop mic (test before hardware)
#   ./asr.sh start --esp32    server + live windows only; ESP32 connects to <laptop-ip>:8766
#   ./asr.sh stop             stop server and laptop mic (windows stay open and reconnect)
#   ./asr.sh restart          stop, then start again in the same mode as last time
#   ./asr.sh status           show what is running
#   ./asr.sh logs             follow the server log (Ctrl+C to leave)
#
# Options for start/restart:  --esp32 | --mic   --windows | --no-windows
# ASR_MODEL=moonshine|whisper picks the post-wake model (default: moonshine).
# MIC_DEVICE=<index> pins the laptop mic (see '.venv/bin/python test_client.py --list');
# by default the input with the strongest signal is used.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
ASR_MODEL="${ASR_MODEL:-moonshine}"
LOGS=logs
SERVER_LOG=$LOGS/server.log
MIC_LOG=$LOGS/mic.log
SERVER_PID=$LOGS/server.pid
MIC_PID=$LOGS/mic.pid
MODE_FILE=$LOGS/mode
mkdir -p "$LOGS"

port_pid() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }
alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

stop_pid() {
  local pid=$1 name=$2
  alive "$pid" || return 0
  kill "$pid" 2>/dev/null
  for _ in $(seq 20); do alive "$pid" || { echo "[-] Stopped $name (pid $pid)"; return 0; }; sleep 0.25; done
  kill -9 "$pid" 2>/dev/null && echo "[-] Force-stopped $name (pid $pid)"
}

open_window() {
  local url="file://$PWD/dashboard/$1"
  if command -v chromium >/dev/null; then
    setsid chromium --new-window --app="$url" >/dev/null 2>&1 &
  elif command -v firefox >/dev/null; then
    setsid firefox --new-window "$url" >/dev/null 2>&1 &
  else
    xdg-open "$url" >/dev/null 2>&1 &
  fi
}

do_stop() {
  stop_pid "$(cat "$MIC_PID" 2>/dev/null)" "laptop mic"
  for pid in $(pgrep -f "$PY test_client.py" 2>/dev/null); do stop_pid "$pid" "laptop mic"; done
  stop_pid "$(cat "$SERVER_PID" 2>/dev/null)" "server"
  local stray; stray=$(port_pid 8766)
  [[ -n "$stray" ]] && stop_pid "$stray" "server on :8766"
  rm -f "$MIC_PID" "$SERVER_PID"
  if [[ -n "$(port_pid 8766)$(port_pid 8767)" ]]; then
    echo "[!] Ports 8766/8767 are still busy:"; ss -ltnp | grep -E ':(8766|8767) '; return 1
  fi
  echo "[+] Everything stopped."
}

do_start() {
  local mode=$1 windows=$2
  if [[ -n "$(port_pid 8766)$(port_pid 8767)" ]]; then
    echo "[!] Already running (ports 8766/8767 in use). Use './asr.sh restart' or './asr.sh stop'."; return 1
  fi
  [[ "$ASR_MODEL" == "ask" ]] && { echo "[!] ASR_MODEL=ask needs a terminal; run '$PY server.py' directly instead."; return 1; }
  echo "$mode" > "$MODE_FILE"

  echo "[*] Starting server (post-wake ASR: $ASR_MODEL)..."
  : > "$SERVER_LOG"
  setsid nohup "$PY" -u server.py --asr "$ASR_MODEL" >> "$SERVER_LOG" 2>&1 < /dev/null &
  echo $! > "$SERVER_PID"
  local pid; pid=$(cat "$SERVER_PID")
  until grep -q "Ready and waiting" "$SERVER_LOG"; do
    alive "$pid" || { echo "[!] Server failed to start. Last lines of $SERVER_LOG:"; tail -20 "$SERVER_LOG"; rm -f "$SERVER_PID"; return 1; }
    sleep 0.5
  done
  echo "[+] Server ready (pid $pid): audio on :8766, live feed on :8767"

  if [[ "$windows" == yes ]]; then
    open_window server.html; sleep 1; open_window asr.html
    echo "[+] Opened Server and ASR windows"
  fi

  if [[ "$mode" == mic ]]; then
    : > "$MIC_LOG"
    setsid nohup "$PY" -u test_client.py ${MIC_DEVICE:+--device "$MIC_DEVICE"} >> "$MIC_LOG" 2>&1 < /dev/null &
    echo $! > "$MIC_PID"
    for _ in $(seq 30); do
      grep -q "Connected!" "$MIC_LOG" && break
      alive "$(cat "$MIC_PID")" || break
      sleep 0.5
    done
    if grep -q "Connected!" "$MIC_LOG"; then
      echo "[+] Laptop mic streaming: $(grep -m1 'Microphone:' "$MIC_LOG" | sed 's/^\[\*\] //')"
      echo "    Say \"Khus Khus\", then your command."
    else
      echo "[!] Laptop mic client failed. Last lines of $MIC_LOG:"; grep -v '^Expression' "$MIC_LOG" | tail -10
    fi
  else
    echo "[*] ESP32 mode: point the ESP32 at one of $(hostname -I | xargs) on port 8766"
  fi
  echo "    Stop with './asr.sh stop'."
}

do_status() {
  local s m; s=$(port_pid 8766); m=$(cat "$MIC_PID" 2>/dev/null)
  if [[ -n "$s" ]]; then echo "[+] Server: running (pid $s), mode: $(cat "$MODE_FILE" 2>/dev/null || echo '?')"; else echo "[-] Server: stopped"; fi
  if alive "$m"; then echo "[+] Laptop mic: streaming (pid $m)"; else echo "[-] Laptop mic: not running"; fi
  local peers; peers=$(ss -tn 2>/dev/null | awk '$4 ~ /:8766$/ {print $5}' | xargs)
  echo "    Audio clients on :8766: ${peers:-none}"
}

cmd=${1:-}; shift || true
mode="" windows=""
for arg in "$@"; do
  case "$arg" in
    --esp32) mode=esp32 ;;
    --mic) mode=mic ;;
    --windows) windows=yes ;;
    --no-windows) windows=no ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

case "$cmd" in
  start)   do_start "${mode:-mic}" "${windows:-yes}" ;;
  stop)    do_stop ;;
  restart) do_stop && do_start "${mode:-$(cat "$MODE_FILE" 2>/dev/null || echo mic)}" "${windows:-no}" ;;
  status)  do_status ;;
  logs)    tail -n 40 -f "$SERVER_LOG" ;;
  *)       sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
