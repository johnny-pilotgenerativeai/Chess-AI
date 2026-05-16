import chess
import ollama
import time
import re
import random

# ─────────────────────────────────────────────
#  MODEL CONFIGURATION
#  Change these to any model you have in Ollama
#  Run 'ollama list' to see available models
# ─────────────────────────────────────────────
AI1_MODEL = "llama3.2:3b"   # White
AI2_MODEL = "llama3.2:3b"   # Black
# ─────────────────────────────────────────────


def algebraic_to_uci(board, alg):
    """Try to convert algebraic notation (Nf3, e4, O-O) to UCI."""
    alg = alg.strip()
    try:
        move = board.parse_san(alg)
        if move in board.legal_moves:
            return move.uci()
    except:
        pass
    for move in board.legal_moves:
        try:
            san = board.san(move).replace("+", "").replace("#", "").replace("x", "")
            clean = alg.replace("+", "").replace("#", "").replace("x", "")
            if san == clean:
                return move.uci()
        except:
            pass
    return None


def extract_move(board, text):
    """
    Extract a valid UCI move from text.
    Requires Move( or Move: prefix — prevents matching natural language like 'your move b8c6'.
    """
    if not text:
        return None, None

    # Block literal placeholder "Move(uci)"
    if re.search(r"\bMove\s*[:(]\s*uci\s*\)?", text, re.IGNORECASE):
        return None, None

    # UCI format — delimiter is required (not optional) to avoid natural language matches
    uci_match = re.search(r"\bMove\s*[:(]\s*([a-h][1-8][a-h][1-8][qrbn]?)\)?", text, re.IGNORECASE)
    if uci_match:
        potential = uci_match.group(1).lower()
        try:
            move = chess.Move.from_uci(potential)
            if move in board.legal_moves:
                return move, None
            else:
                print(f"  System: '{potential}' is not a legal move — treating as fake.")
                return None, potential
        except:
            pass

    # Fuzzy rescue — algebraic: Move(Nf3), Move(e4), Move(O-O)
    alg_match = re.search(
        r"\bMove\s*[:(]\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)\)?",
        text, re.IGNORECASE
    )
    if alg_match:
        alg = alg_match.group(1)
        uci = algebraic_to_uci(board, alg)
        if uci:
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    print(f"  System: Rescued algebraic '{alg}' -> UCI '{uci}'")
                    return move, None
            except:
                pass
        return None, alg

    return None, None


def get_board_perspective(color):
    """Return the board coordinate orientation for each colour."""
    if color == "White":
        return (
            "Files (columns) left to right: A B C D E F G H\n"
            "Ranks (rows) bottom to top:    1 2 3 4 5 6 7 8\n"
            "Your pieces start on ranks 1 and 2 (bottom).\n"
            "Opponent pieces start on ranks 7 and 8 (top)."
        )
    else:
        return (
            "Files (columns) left to right: H G F E D C B A\n"
            "Ranks (rows) bottom to top:    8 7 6 5 4 3 2 1\n"
            "Your pieces start on ranks 8 and 7 (bottom from your view).\n"
            "Opponent pieces start on ranks 2 and 1 (top from your view)."
        )


def get_ai_message(model, ai_color_name, chat_log, move_history, board, is_active_turn, nudge=False, bad_move=None):
    """Call Ollama with the specified model. Move history and chat are passed separately."""
    legal_moves = [move.uci() for move in board.legal_moves]
    examples = ", ".join(f"Move({m})" for m in legal_moves[:3])
    perspective = get_board_perspective(ai_color_name)

    recent_chat = "\n".join(chat_log[-6:]) if chat_log else "No messages yet."
    move_history_str = "\n".join(move_history) if move_history else "No moves played yet."

    move_warning = (
        "IMPORTANT: The moves listed in 'Moves already played' have ALL been played. "
        "Do NOT attempt to play any of them again. "
        "You MUST choose from 'Your legal moves' below."
    )

    if nudge or bad_move:
        reason = (
            f"Your last attempt '{bad_move}' was the wrong format."
            if bad_move else
            "You have been chatting too long without moving."
        )
        prompt = f"""You are a chess AI playing as {ai_color_name}. {reason}
YOU MUST MAKE YOUR MOVE NOW.

Your board perspective:
{perspective}

Moves already played (DO NOT REPEAT THESE):
{move_history_str}

{move_warning}

Your legal moves for THIS turn: {', '.join(legal_moves)}
Board FEN (current position): {board.fen()}

UCI FORMAT ONLY — source square then destination square:
  WRONG: Move(Nf3)  WRONG: Move(e4)  WRONG: Move(uci)
  RIGHT: Move({legal_moves[0]})  RIGHT: Move({legal_moves[1]})

Respond with ONE sentence and your move, e.g: "Taking control now. Move({legal_moves[0]})"
"""

    elif is_active_turn:
        prompt = f"""You are a chess AI playing as {ai_color_name} in a chess match against another AI.
Chat with your opponent. When ready to move, include your move in your message.

Your board perspective:
{perspective}

Moves already played (DO NOT REPEAT THESE):
{move_history_str}

{move_warning}

Your legal moves for THIS turn: {', '.join(legal_moves)}
Board FEN (current position): {board.fen()}

UCI FORMAT ONLY — source square then destination square:
  WRONG: Move(Nf3)  WRONG: Move(e4)  WRONG: Move(uci)
  RIGHT examples: {examples}

Recent conversation:
{recent_chat}

Your response (chat and/or your move):
"""

    else:
        prompt = f"""You are a chess AI playing as {ai_color_name} in a chess match.
It is NOT your turn. Write 1-2 sentences replying to your opponent. Do NOT use Move() syntax.

Your board perspective:
{perspective}

Moves already played:
{move_history_str}

Recent conversation:
{recent_chat}

Your response:
"""

    try:
        response = ollama.chat(model=model, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"  System: Ollama error ({model}) — {e}")
        return ""


def render_board(board, is_surrender=False, surrender_color=None):
    """Render the board with coordinates and borders."""
    board_str = str(board)
    if is_surrender:
        king_to_kill = 'K' if surrender_color == 'White' else 'k'
        board_str = board_str.replace(king_to_kill, 'X')
        print("\n" + "X" * 10 + " SURRENDERED " + "X" * 10)
    elif board.is_check():
        check_king = 'K' if board.turn == chess.WHITE else 'k'
        board_str = board_str.replace(check_king, '!')
        print("\n" + "!" * 10 + " CHECK! " + "!" * 10)

    rows = board_str.split('\n')
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    files = "A B C D E F G H"
    border = "  +-----------------+"

    print(f"{files}  Black <--")
    print(border)
    for rank_label, row in zip(ranks, rows):
        print(f"{rank_label} | {row} |")
    print(border)
    print(f"{files}  White <--")


def print_turn_banner(board, ai1_model, ai2_model):
    if board.turn == chess.WHITE:
        print(f"\n{'#' * 50}")
        print(f"  >>> WHITE's TURN  [Turn {board.fullmove_number}]  (AI #1 — {ai1_model}) <<<")
        print(f"{'#' * 50}")
    else:
        print(f"\n{'~' * 50}")
        print(f"  >>> BLACK's TURN  [Turn {board.fullmove_number}]  (AI #2 — {ai2_model}) <<<")
        print(f"{'~' * 50}")


def play_ai_vs_ai():
    board = chess.Board()
    chat_log = []
    move_history = []
    MAX_ACTIVE_MESSAGES = 3

    print("=" * 50)
    print("       AI vs AI - CHESS MATCH")
    print("=" * 50)
    print(f"  AI #1 (White) = {AI1_MODEL}")
    print(f"  AI #2 (Black) = {AI2_MODEL}")
    print("=" * 50)

    print_turn_banner(board, AI1_MODEL, AI2_MODEL)
    render_board(board)
    time.sleep(4)

    while not board.is_game_over():
        active_color   = "White" if board.turn == chess.WHITE else "Black"
        opponent_color = "Black" if active_color == "White" else "White"
        active_num     = "#1" if active_color == "White" else "#2"
        opponent_num   = "#2" if active_color == "White" else "#1"
        active_model   = AI1_MODEL if active_color == "White" else AI2_MODEL
        opponent_model = AI2_MODEL if active_color == "White" else AI1_MODEL

        move_made = False
        active_messages = 0

        while not move_made and not board.is_game_over():
            nudge = active_messages >= MAX_ACTIVE_MESSAGES

            # --- Active player speaks ---
            print(f"\nAI {active_num} ({active_color} — {active_model}) is typing...")
            time.sleep(1)

            message = get_ai_message(active_model, active_color, chat_log, move_history, board, True, nudge=nudge)

            # Retry once if blank
            if not message:
                print(f"  System: AI {active_num} returned blank. Retrying with nudge...")
                message = get_ai_message(active_model, active_color, chat_log, move_history, board, True, nudge=True)

            # Force random if still blank
            if not message:
                print(f"  System: AI {active_num} still blank. Forcing random move.")
                move = random.choice(list(board.legal_moves))
                board.push(move)
                entry = f"Turn {board.fullmove_number} | AI {active_num} ({active_color}): {move.uci()} [forced random]"
                move_history.append(entry)
                print(f"\n  *** AI {active_num} forced random -> {move.uci()} ***")
                move_made = True
                break

            print(f"AI {active_num} ({active_color}): {message}")
            chat_log.append(f"AI {active_num} ({active_color}): {message}")

            # Try to extract a move
            move, bad_alg = extract_move(board, message)

            # Correction attempt if bad format detected
            if bad_alg and not move:
                print(f"  System: Bad move format '{bad_alg}'. Asking for correction...")
                correction = get_ai_message(active_model, active_color, chat_log, move_history, board, True, bad_move=bad_alg)
                if correction:
                    print(f"AI {active_num} ({active_color}) [corrected]: {correction}")
                    chat_log.append(f"AI {active_num} ({active_color}): {correction}")
                    move, _ = extract_move(board, correction)

            # Force random if nudged and still no valid move
            if nudge and not move:
                print(f"  System: AI {active_num} exceeded chat limit. Forcing random move.")
                move = random.choice(list(board.legal_moves))

            if move:
                board.push(move)
                entry = f"Turn {board.fullmove_number} | AI {active_num} ({active_color}): {move.uci()}"
                move_history.append(entry)
                print(f"\n  *** AI {active_num} ({active_color}) plays -> {move.uci()} ***")
                move_made = True
                time.sleep(2)

                if not board.is_game_over():
                    print_turn_banner(board, AI1_MODEL, AI2_MODEL)
                    render_board(board)
                    time.sleep(4)
                break

            active_messages += 1

            # --- Opponent gets ONE brief reply ---
            print(f"\nAI {opponent_num} ({opponent_color} — {opponent_model}) is typing...")
            time.sleep(1)
            opp_message = get_ai_message(opponent_model, opponent_color, chat_log, move_history, board, False)

            if opp_message:
                print(f"AI {opponent_num} ({opponent_color}): {opp_message}")
                chat_log.append(f"AI {opponent_num} ({opponent_color}): {opp_message}")
            else:
                print(f"AI {opponent_num} ({opponent_color}): ...")

            time.sleep(2)

    # Final board
    print("\n" + "=" * 50)
    render_board(board)
    result = board.result()
    if result == "1-0":
        print(f"\nGAME OVER! AI #1 ({AI1_MODEL}) as White WINS!")
    elif result == "0-1":
        print(f"\nGAME OVER! AI #2 ({AI2_MODEL}) as Black WINS!")
    else:
        print("\nGAME OVER! It's a DRAW!")
    print("=" * 50)


if __name__ == "__main__":
    play_ai_vs_ai()
