from flask import Flask, Response, render_template_string
import chess
import ollama
import threading
import queue
import json
import time
import re
import random

app = Flask(__name__)

# ─────────────────────────────────────────────
#  MODEL CONFIGURATION
# ─────────────────────────────────────────────
AI1_MODEL = "qwen2.5:1.5b"   # White
AI2_MODEL = "qwen2.5:1.5b"   # Black

# Seconds to wait for Ollama before giving up and forcing a random move
OLLAMA_TIMEOUT = 45
MAX_ACTIVE_MESSAGES = 2
# ─────────────────────────────────────────────

_subscribers  = []
_sub_lock     = threading.Lock()
_game_running = False
_current_fen  = chess.STARTING_FEN
_current_turn = "White"


def broadcast(event_type, **kwargs):
    payload = json.dumps({"type": event_type, **kwargs})
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


# ── Chess helpers ──────────────────────────────────────────────
def algebraic_to_uci(board, alg):
    alg = alg.strip()
    try:
        move = board.parse_san(alg)
        if move in board.legal_moves:
            return move.uci()
    except Exception:
        pass
    for move in board.legal_moves:
        try:
            san   = board.san(move).replace("+","").replace("#","").replace("x","")
            clean = alg.replace("+","").replace("#","").replace("x","")
            if san == clean:
                return move.uci()
        except Exception:
            pass
    return None


# Move patterns tried in order — most strict first, loosest last
_MOVE_PATTERNS = [
    # 1. Correct format:  Move:(e2e4)
    (r"\bMove\s*:\s*\(\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\)", "correct"),
    # 2. Parens only:     Move(e2e4)
    (r"\bMove\s*\(\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\)",      "parens"),
    # 3. Colon space:     Move: e2e4  or  move: e2e4
    (r"\bMove\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b",            "colon"),
    # 4. Bare word:       move e2e4
    (r"\bMove\s+([a-h][1-8][a-h][1-8][qrbn]?)\b",                 "bare"),
    # 5. Quoted:          move = "e2e4"  or  move="e2e4"
    (r'\bMove\s*=\s*["\']([a-h][1-8][a-h][1-8][qrbn]?)["\']\b', "quoted"),
    # 6. Standalone UCI anywhere in the message (last resort)
    (r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b",                         "standalone"),
]

_ALG_PATTERNS = [
    (r"\bMove\s*:\s*\(\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)\s*\)", "alg-correct"),
    (r"\bMove\s*\(\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)\s*\)",      "alg-parens"),
    (r"\bMove\s*:\s*([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)\b",              "alg-colon"),
]


def try_uci(board, text):
    """Try all UCI patterns, return (move, fmt_used) or (None, None)."""
    # Block literal placeholder
    if re.search(r"\bMove\s*:?\s*\(?\s*uci\s*\)?", text, re.IGNORECASE):
        return None, None
    for pattern, fmt in _MOVE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            candidate = m.group(1).lower()
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move, fmt
            except Exception:
                pass
    return None, None


def try_alg(board, text):
    """Try algebraic rescue patterns."""
    for pattern, fmt in _ALG_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            uci = algebraic_to_uci(board, m.group(1))
            if uci:
                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        return move, m.group(1)
                except Exception:
                    pass
            return None, m.group(1)
    return None, None


def extract_move(board, text):
    """
    Soft call chain — tries every reasonable move syntax.
    Logs when a non-standard format is used and nudges toward Move:(e2e4).
    """
    if not text:
        return None, None

    # Try UCI patterns
    move, fmt = try_uci(board, text)
    if move:
        legal = [m.uci() for m in board.legal_moves]
        if fmt != "correct":
            broadcast("system",
                msg=f"Accepted '{fmt}' format — please use Move:({move.uci()}) next time.")
        return move, None

    # Try algebraic rescue
    move, bad_alg = try_alg(board, text)
    if move:
        broadcast("system", msg=f"Rescued algebraic '{bad_alg}' -> '{move.uci()}'")
        return move, None
    if bad_alg:
        return None, bad_alg

    return None, None


def ollama_call(model, prompt):
    """Call Ollama with a hard timeout. Returns empty string on timeout/error."""
    result = [""]
    error  = [None]

    def _call():
        try:
            r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            result[0] = r["message"]["content"].strip()
        except Exception as e:
            error[0] = str(e)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=OLLAMA_TIMEOUT)

    if t.is_alive():
        broadcast("system", msg=f"Ollama timeout ({OLLAMA_TIMEOUT}s) — forcing random move.")
        return ""
    if error[0]:
        broadcast("system", msg=f"Ollama error: {error[0]}")
        return ""
    return result[0]


def get_ai_message(model, ai_color, chat_log, move_history, board, is_active, nudge=False, bad_move=None):
    """Lean prompts — short enough for 1B models to handle quickly."""
    legal = [m.uci() for m in board.legal_moves]
    # Only show last 3 moves and last 3 chat lines to keep prompts short
    recent_moves = move_history[-3:] if move_history else []
    recent_chat  = chat_log[-3:]     if chat_log     else []

    # Pick 3 example moves to show
    ex = " | ".join(legal[:3])

    if nudge or bad_move:
        reason = f"Wrong format: {bad_move}." if bad_move else "Too much chat, move now."
        # Pick a random legal move to use as a concrete example
        example = random.choice(legal)
        prompt = (
            f"Chess AI playing as {ai_color}. {reason}\n"
            f"MAKE YOUR MOVE NOW.\n"
            f"ONLY accepted format: Move:({example}) — colon then parentheses around the UCI move.\n"
            f"Examples: Move:({legal[0]})  Move:({legal[1] if len(legal)>1 else legal[0]})\n"
            f"Legal moves: {', '.join(legal)}\n"
            f"FEN: {board.fen()}\n"
            f"Reply: one sentence + Move:(your_choice_here)"
        )

    elif is_active:
        moves_str = ", ".join(recent_moves) if recent_moves else "none yet"
        chat_str  = " | ".join(recent_chat) if recent_chat  else "none"
        prompt = (
            f"You are a chess AI playing as {ai_color}.\n"
            f"Recent moves: {moves_str}\n"
            f"Recent chat: {chat_str}\n"
            f"FEN: {board.fen()}\n"
            f"Legal moves this turn: {', '.join(legal)}\n"
            f"Format: Move:(e2e4) — examples: {ex}\n"
            f"Chat briefly then make your move with Move:(source+dest)."
        )

    else:
        chat_str = " | ".join(recent_chat) if recent_chat else "none"
        prompt = (
            f"You are a chess AI playing as {ai_color}. It is NOT your turn.\n"
            f"Recent chat: {chat_str}\n"
            f"Reply in 1 sentence — trash talk, taunt, or react to their last move.\n"
            f"STRICT RULES:\n"
            f"- Do NOT use Move:() syntax.\n"
            f"- Do NOT suggest, provide, or hint any square names or move coordinates to your opponent.\n"
            f"- You may bluff about your OWN future plans to trick them, but never give real advice.\n"
            f"- Never say things like 'you should move to X' or 'try playing X'."
        )

    return ollama_call(model, prompt)


# ── Game loop ──────────────────────────────────────────────────
def run_game():
    global _game_running, _current_fen, _current_turn
    board        = chess.Board()
    chat_log     = []
    move_history = []
    _game_running = True

    broadcast("system", msg="Game started!")
    broadcast("board", fen=board.fen(), turn="White", move_number=1,
              last_move=None, in_check=False)

    while not board.is_game_over():
        active_color   = "White" if board.turn == chess.WHITE else "Black"
        opponent_color = "Black" if active_color == "White"   else "White"
        active_num     = "#1"    if active_color == "White"   else "#2"
        opponent_num   = "#2"    if active_color == "White"   else "#1"
        active_model   = AI1_MODEL if active_color == "White" else AI2_MODEL
        opponent_model = AI2_MODEL if active_color == "White" else AI1_MODEL

        _current_turn = active_color
        broadcast("turn", color=active_color, num=active_num,
                  model=active_model, move_number=board.fullmove_number)

        move_made       = False
        active_messages = 0

        while not move_made and not board.is_game_over():
            nudge = active_messages >= MAX_ACTIVE_MESSAGES

            broadcast("typing", num=active_num, color=active_color, model=active_model)

            message = get_ai_message(active_model, active_color, chat_log,
                                     move_history, board, True, nudge=nudge)

            # Blank or timeout — force random immediately
            if not message:
                move = random.choice(list(board.legal_moves))
                board.push(move)
                _current_fen = board.fen()
                move_history.append(f"T{board.fullmove_number} {active_color}: {move.uci()} [random]")
                broadcast("system",  msg=f"AI {active_num} timed out — random move: {move.uci()}")
                broadcast("move",    num=active_num, color=active_color, uci=move.uci())
                broadcast("board",   fen=board.fen(),
                          turn="White" if board.turn == chess.WHITE else "Black",
                          move_number=board.fullmove_number,
                          last_move=move.uci(), in_check=board.is_check())
                move_made = True
                break

            broadcast("chat", num=active_num, color=active_color,
                      message=message, model=active_model)
            chat_log.append(f"AI{active_num}({active_color}): {message}")

            move, bad_alg = extract_move(board, message)

            # One correction attempt
            if bad_alg and not move:
                broadcast("system", msg=f"Bad format '{bad_alg}', correcting...")
                correction = get_ai_message(active_model, active_color, chat_log,
                                            move_history, board, True, bad_move=bad_alg)
                if correction:
                    broadcast("chat", num=active_num, color=active_color,
                              message=f"[fix] {correction}", model=active_model)
                    chat_log.append(f"AI{active_num}({active_color}): {correction}")
                    move, _ = extract_move(board, correction)

            # Force random if nudged and still nothing
            if nudge and not move:
                broadcast("system", msg=f"AI {active_num} exceeded limit — random move.")
                move = random.choice(list(board.legal_moves))

            if move:
                board.push(move)
                _current_fen = board.fen()
                move_history.append(f"T{board.fullmove_number} {active_color}: {move.uci()}")
                broadcast("move",  num=active_num, color=active_color, uci=move.uci())
                broadcast("board", fen=board.fen(),
                          turn="White" if board.turn == chess.WHITE else "Black",
                          move_number=board.fullmove_number,
                          last_move=move.uci(), in_check=board.is_check())
                move_made = True
                time.sleep(1)
                break

            active_messages += 1

            # Opponent gets one brief reply — with timeout too
            broadcast("typing", num=opponent_num, color=opponent_color, model=opponent_model)
            opp = get_ai_message(opponent_model, opponent_color, chat_log,
                                 move_history, board, False)
            if opp:
                broadcast("chat", num=opponent_num, color=opponent_color,
                          message=opp, model=opponent_model)
                chat_log.append(f"AI{opponent_num}({opponent_color}): {opp}")
            time.sleep(1)

    result = board.result()
    if   result == "1-0": msg = f"AI #1 ({AI1_MODEL}) White WINS!"
    elif result == "0-1": msg = f"AI #2 ({AI2_MODEL}) Black WINS!"
    else:                 msg = "It's a DRAW!"
    broadcast("gameover", result=result, message=msg)
    _game_running = False


# ── Flask routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, ai1=AI1_MODEL, ai2=AI2_MODEL)


@app.route("/start", methods=["POST"])
def start_game():
    global _game_running
    if not _game_running:
        threading.Thread(target=run_game, daemon=True).start()
        return {"status": "started"}
    return {"status": "already_running"}


@app.route("/pi-status")
def pi_status():
    """Check Pi voltage and throttle state via vcgencmd."""
    import subprocess
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=3
        )
        # Returns e.g. "throttled=0x0" or "throttled=0x50005"
        raw = result.stdout.strip()
        val_str = raw.split("=")[-1]
        val = int(val_str, 16)
        under_voltage_now  = bool(val & 0x1)
        throttled_now      = bool(val & 0x4)
        under_voltage_ever = bool(val & 0x10000)
        throttled_ever     = bool(val & 0x40000)
        return {
            "raw": raw,
            "under_voltage_now":  under_voltage_now,
            "throttled_now":      throttled_now,
            "under_voltage_ever": under_voltage_ever,
            "throttled_ever":     throttled_ever,
            "warn": under_voltage_now or throttled_now
        }
    except Exception as e:
        return {"error": str(e), "warn": False}


@app.route("/stream")
def stream():
    def event_stream():
        q = queue.Queue(maxsize=200)
        with _sub_lock:
            _subscribers.append(q)
        init = json.dumps({"type": "board", "fen": _current_fen,
                           "turn": _current_turn, "move_number": 1,
                           "last_move": None, "in_check": False})
        yield f"data: {init}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield 'data: {"type":"ping"}\n\n'
        except GeneratorExit:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── HTML ───────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Chess — {{ ai1 }} vs {{ ai2 }}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0d0d1a; color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}
header {
    background: #12122a; padding: 10px 24px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #2a2a4a; flex-shrink: 0;
}
header h1 { font-size: 1.1rem; color: #fff; letter-spacing: 1px; }
.models { font-size: 0.82rem; color: #888; }
.models .w { color: #ffffff; font-weight: bold; }
.models .b { color: #64dfdf; font-weight: bold; }
main { display: flex; flex: 1; overflow: hidden; }
.board-panel {
    width: 50%; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 16px; background: #0f0f20; gap: 10px; flex-shrink: 0;
}
#turn-badge {
    padding: 5px 18px; border-radius: 20px; font-size: 0.85rem;
    font-weight: 600; letter-spacing: 0.5px; transition: all 0.4s;
    background: #1e1e3a; color: #888; border: 1px solid #333;
}
#turn-badge.white-turn { background: #2a2a4a; color: #fff;    border-color: #aaa; }
#turn-badge.black-turn { background: #092830; color: #64dfdf; border-color: #64dfdf; }
#turn-badge.game-over  { background: #2a0808; color: #ff6b6b; border-color: #ff6b6b; }
.board-outer {
    --board-size: min(440px, 42vw, 70vh);
    --sq-size: calc(var(--board-size) / 8);
    display: flex; align-items: flex-start; gap: 4px;
}
.rank-labels { display: flex; flex-direction: column; width: 16px; flex-shrink: 0; }
.rank-labels span {
    height: var(--sq-size); display: flex; align-items: center;
    justify-content: center; font-size: 0.68rem; color: #666; line-height: 1;
}
.board-col { display: flex; flex-direction: column; }
#board-grid {
    display: grid;
    grid-template-columns: repeat(8, var(--sq-size));
    grid-template-rows:    repeat(8, var(--sq-size));
    width: var(--board-size); height: var(--board-size);
    border: 2px solid #333; border-radius: 2px; overflow: hidden; flex-shrink: 0;
}
.file-labels { display: flex; margin-top: 4px; width: var(--board-size); }
.file-labels span {
    width: var(--sq-size); font-size: 0.68rem; color: #666;
    text-align: center; flex-shrink: 0;
}
.sq {
    width: var(--sq-size); height: var(--sq-size);
    display: flex; align-items: center; justify-content: center;
    position: relative; transition: background 0.2s; overflow: hidden;
}
.sq.light { background: #eeeed2; }
.sq.dark  { background: #769656; }
.sq.hilite-from { background: rgba(246,246,105,0.75) !important; }
.sq.hilite-to   { background: rgba(246,246,105,0.95) !important; }
.sq.in-check    { background: rgba(220,50,50,0.85)   !important; }
.piece {
    font-size: calc(var(--sq-size) * 0.72);
    line-height: 1; user-select: none; pointer-events: none;
}
.wp { color: #fff;    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.7)); }
.bp { color: #1a1a1a; filter: drop-shadow(0 1px 2px rgba(255,255,255,0.25)); }
.chat-panel {
    width: 50%; display: flex; flex-direction: column;
    background: #0d0d1a; border-left: 1px solid #1e1e3a; overflow: hidden;
}
.chat-header {
    padding: 10px 16px; background: #12122a; font-size: 0.82rem;
    color: #666; border-bottom: 1px solid #1e1e3a; flex-shrink: 0;
}
#chat-log {
    flex: 1; overflow-y: auto; padding: 10px 12px;
    display: flex; flex-direction: column; gap: 6px; scroll-behavior: smooth;
}
#chat-log::-webkit-scrollbar { width: 4px; }
#chat-log::-webkit-scrollbar-track { background: #0d0d1a; }
#chat-log::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 4px; }
.msg {
    padding: 7px 11px; border-radius: 7px; font-size: 0.84rem;
    line-height: 1.45; word-wrap: break-word; max-width: 92%;
    animation: fadeIn 0.25s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(4px)}to{opacity:1} }
.msg.ai1      { background:#1a1a35; border-left:3px solid #ffffff; color:#e8e8e8; align-self:flex-start; }
.msg.ai2      { background:#081e25; border-left:3px solid #64dfdf; color:#a8f0f0; align-self:flex-end; }
.msg.system   { background:#1e1800; border-left:3px solid #f9c74f; color:#f9c74f; align-self:center; font-size:0.78rem; font-style:italic; max-width:85%; }
.msg.move-ann { background:#0d2010; border-left:3px solid #90ee90; color:#90ee90; align-self:center; font-weight:600; }
.msg.gameover { background:#2a0808; border-left:3px solid #ff6b6b; color:#ff6b6b; align-self:center; font-weight:700; font-size:0.95rem; text-align:center; }
.msg-label { font-size: 0.68rem; color: #555; margin-bottom: 2px; }
.typing-row {
    padding: 3px 12px; font-size: 0.78rem; color: #444;
    font-style: italic; flex-shrink: 0; min-height: 20px;
}
.chat-footer {
    padding: 10px 14px; border-top: 1px solid #1e1e3a;
    display: flex; align-items: center; gap: 10px;
    background: #0f0f20; flex-shrink: 0;
}
#start-btn {
    background: #0d1e35; color: #64dfdf; border: 1px solid #64dfdf;
    padding: 7px 18px; border-radius: 6px; cursor: pointer;
    font-size: 0.85rem; transition: all 0.2s; white-space: nowrap;
}
#start-btn:hover:not(:disabled) { background: #64dfdf; color: #0d0d1a; }
#start-btn:disabled { opacity: 0.35; cursor: not-allowed; }
#status-txt { font-size: 0.78rem; color: #444; }
#disconnect-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.93);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999; animation: fadeIn 0.3s ease;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}
#disconnect-box {
  background: #1a0808; border: 2px solid #ff6b6b; border-radius: 12px;
  padding: 32px 40px; text-align: center; max-width: 360px;
  box-shadow: 0 0 60px rgba(255,50,50,0.3);
}
#disconnect-icon  { font-size: 2.5rem; margin-bottom: 8px; }
#disconnect-title { font-size: 1.3rem; font-weight: 700; color: #ff6b6b; margin-bottom: 8px; }
#disconnect-msg   { font-size: 0.85rem; color: #ccc; margin-bottom: 12px; }
#disconnect-retry { font-size: 0.78rem; color: #888; font-style: italic; }
#power-warn {
  display: none; position: fixed; top: 12px; right: 12px;
  background: #2a1500; border: 2px solid #ff9500; border-radius: 8px;
  padding: 6px 14px; color: #ff9500; font-size: 0.82rem; font-weight: 600;
  z-index: 1000; animation: fadeIn 0.3s ease; cursor: default;
}
#power-warn.visible { display: flex; align-items: center; gap: 6px; }
</style>
</head>
<body>
<div id="power-warn" title="Pi voltage is too low — consider a 27W power supply">
  ⚡ Low Voltage Warning
</div>
<header>
  <h1>♟ AI Chess</h1>
  <div class="models">
    <span class="w">AI #1 White</span> = {{ ai1 }}
    &nbsp;·&nbsp;
    <span class="b">AI #2 Black</span> = {{ ai2 }}
  </div>
</header>
<main>
  <div class="board-panel">
    <div id="turn-badge">Waiting to start…</div>
    <div class="board-outer">
      <div class="rank-labels">
        <span>8</span><span>7</span><span>6</span><span>5</span>
        <span>4</span><span>3</span><span>2</span><span>1</span>
      </div>
      <div class="board-col">
        <div id="board-grid"></div>
        <div class="file-labels">
          <span>a</span><span>b</span><span>c</span><span>d</span>
          <span>e</span><span>f</span><span>g</span><span>h</span>
        </div>
      </div>
    </div>
  </div>
  <div class="chat-panel">
    <div class="chat-header">Live Game Chat</div>
    <div id="chat-log"></div>
    <div class="typing-row" id="typing-row"></div>
    <div class="chat-footer">
      <button id="start-btn" onclick="startGame()">▶ Start Game</button>
      <span id="status-txt">Press Start to begin</span>
    </div>
  </div>
</main>
<script>
const PIECES = {
  K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',
  k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'
};
function renderBoard(fen, lastMove, inCheck) {
  const [pos, turn] = fen.split(' ');
  const rows = pos.split('/');
  let fromSq = null, toSq = null;
  if (lastMove && lastMove.length >= 4) {
    fromSq = lastMove.slice(0,2); toSq = lastMove.slice(2,4);
  }
  const kingChar = (turn === 'w') ? 'K' : 'k';
  let html = '';
  for (let rowIdx = 0; rowIdx < 8; rowIdx++) {
    const rank = 8 - rowIdx; let file = 0;
    for (const ch of rows[rowIdx]) {
      if (ch >= '1' && ch <= '9') {
        for (let i = 0; i < parseInt(ch); i++) {
          const sqName = 'abcdefgh'[file] + rank;
          const isLight = (rank + file) % 2 !== 0;
          let cls = `sq ${isLight?'light':'dark'}`;
          if (sqName === fromSq) cls += ' hilite-from';
          if (sqName === toSq)   cls += ' hilite-to';
          html += `<div class="${cls}"></div>`; file++;
        }
      } else {
        const sqName = 'abcdefgh'[file] + rank;
        const isLight = (rank + file) % 2 !== 0;
        const isW = ch === ch.toUpperCase();
        let cls = `sq ${isLight?'light':'dark'}`;
        if (sqName === fromSq)          cls += ' hilite-from';
        if (sqName === toSq)            cls += ' hilite-to';
        if (inCheck && ch === kingChar) cls += ' in-check';
        html += `<div class="${cls}"><span class="piece ${isW?'wp':'bp'}">${PIECES[ch]||ch}</span></div>`;
        file++;
      }
    }
  }
  document.getElementById('board-grid').innerHTML = html;
}
function addMsg(type, content, label) {
  if (disconnected) return;  // don't add messages while disconnected
  const log = document.getElementById('chat-log');
  const wrap = document.createElement('div');
  wrap.className = `msg ${type}`;
  if (label) {
    const l = document.createElement('div');
    l.className = 'msg-label'; l.textContent = label; wrap.appendChild(l);
  }
  const t = document.createElement('div');
  t.textContent = content; wrap.appendChild(t);
  log.appendChild(wrap); log.scrollTop = log.scrollHeight;
}
function setTyping(txt) {
  if (disconnected) return;
  document.getElementById('typing-row').textContent = txt;
}
renderBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', null, false);

// ── Pi power status polling ─────────────────
function checkPowerStatus() {
  fetch('/pi-status')
    .then(r => r.json())
    .then(d => {
      const badge = document.getElementById('power-warn');
      if (d.warn) {
        badge.classList.add('visible');
        badge.title = d.raw + ' — Under-voltage or throttling detected. Use a 27W power supply.';
      } else if (d.under_voltage_ever || d.throttled_ever) {
        badge.classList.add('visible');
        badge.style.borderColor = '#f9c74f';
        badge.style.color = '#f9c74f';
        badge.style.background = '#1e1800';
        badge.innerHTML = '⚡ Voltage dip detected (resolved)';
        badge.title = d.raw + ' — Low voltage occurred earlier this session.';
      } else {
        badge.classList.remove('visible');
      }
    })
    .catch(() => {}); // silently ignore if vcgencmd unavailable
}
checkPowerStatus();
setInterval(checkPowerStatus, 15000);  // check every 15s

// ── Connection monitoring ──────────────────────
let lastPing = Date.now();
let disconnected = false;

function showDisconnected() {
  if (disconnected) return;
  disconnected = true;
  const overlay = document.createElement('div');
  overlay.id = 'disconnect-overlay';
  overlay.innerHTML = `
    <div id="disconnect-box">
      <div id="disconnect-icon">⚠</div>
      <div id="disconnect-title">Host Disconnected</div>
      <div id="disconnect-msg">The Raspberry Pi has gone offline or lost power.</div>
      <div id="disconnect-retry">Attempting to reconnect in 5 seconds…</div>
    </div>`;
  document.body.appendChild(overlay);
}

function hideDisconnected() {
  disconnected = false;
  const el = document.getElementById('disconnect-overlay');
  if (el) el.remove();
}

// Show disconnect if no ping in 35s
setInterval(() => {
  if (Date.now() - lastPing > 35000) showDisconnected();
}, 5000);

function connectSSE() {
  const es = new EventSource('/stream');
es.onmessage = function(e) {
  const d = JSON.parse(e.data);
  switch(d.type) {
    case 'board':
      renderBoard(d.fen, d.last_move||null, d.in_check||false);
      if (d.turn) {
        const badge = document.getElementById('turn-badge');
        badge.textContent = `Turn ${d.move_number} — ${d.turn} to move`;
        badge.className   = `${d.turn.toLowerCase()}-turn`;
      }
      break;
    case 'typing':
      setTyping(`AI ${d.num} (${d.color}) is thinking…`);
      break;
    case 'turn': setTyping(''); break;
    case 'chat':
      setTyping('');
      addMsg(d.num==='#1'?'ai1':'ai2', d.message, `AI ${d.num} (${d.color}) — ${d.model}`);
      break;
    case 'move':
      addMsg('move-ann', `♟ AI ${d.num} (${d.color}) played ${d.uci}`);
      break;
    case 'system': addMsg('system', d.msg); break;
    case 'gameover':
      setTyping('');
      addMsg('gameover', `🏁 ${d.message}`);
      document.getElementById('turn-badge').textContent = 'Game Over';
      document.getElementById('turn-badge').className   = 'game-over';
      document.getElementById('start-btn').disabled     = false;
      document.getElementById('status-txt').textContent = 'Game finished';
      break;
    case 'ping': lastPing = Date.now(); break;
  }
};
  return es;
}
connectSSE();

function startGame() {
  fetch('/start', {method:'POST'}).then(r=>r.json()).then(d=>{
    if (d.status==='started') {
      document.getElementById('start-btn').disabled=true;
      document.getElementById('status-txt').textContent='Game running…';
    } else {
      document.getElementById('status-txt').textContent='Already running';
    }
  });
}
window.addEventListener('load', () => startGame());
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("=" * 45)
    print("  AI Chess Flask App")
    print(f"  White : {AI1_MODEL}")
    print(f"  Black : {AI2_MODEL}")
    print(f"  Timeout: {OLLAMA_TIMEOUT}s per move")
    print("=" * 45)
    print("  Open  http://localhost:5000")
    print("=" * 45)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
