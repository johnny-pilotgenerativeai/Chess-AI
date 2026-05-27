# AI Chess: Llama vs Llama 🦙♟️

An open-source chess engine simulation where two fine-tuned Large Language Models (LLMs) play chess against each other. Watch the battle of strategic minds as Llama takes on Llama in a purely algorithmic checkmate showdown.

<p align="center">
  <img src="assets/ChatGPT Image May 18, 2026, 06_38_29 AM.png" alt="Two cute llamas playing chess" width="600">
</p>

---

## 🚀 Features

* **LLM vs LLM Mode:** Fully automated chess games played between two local or API-driven Llama instances.
* **State Validation:** Built-in chess logic validator to ensure models only make legal moves.
* **Prompt Engineering for Strategy:** Custom system prompts that force the models to think ahead and evaluate board states textually before generating a move.
* **Live Visualizer:** A clean web/terminal interface to watch the match play out in real-time.

## 🛠️ Tech Stack

* **Core Engine:** Python 3.10+
* **AI Models:** Llama 3 (via Ollama / Hugging Face)
* **Chess Logic:** `python-chess`
* **UI/Display:** Streamlit (or Terminal-based ASCII)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johnny-pilotgenerativeai/Chess-AI.git
   cd Chess-AI
   ```
    Install dependencies:
```Bash

pip install -r requirements.txt
```
Ensure Ollama/Llama is running locally:
```Bash

ollama run <model>
```
Run the simulation:
```Bash

python3 Chess.py
```
🎮 How it Works

  The game loop initializes a standard 8×8 chess board.

  Llama-White receives the current board state in Forsyth-Edwards Notation (FEN) and a list of legal moves.

  The model processes the prompt and outputs its chosen move in Standard Algebraic Notation (SAN), e.g., e4.

  The engine validates the move. If legal, the board updates.

  Llama-Black receives the updated board, and the cycle repeats until checkmate, stalemate, or draw.

# 🤝 Contributing

Contributions are always welcome! Feel free to open an issue or submit a pull request if you want to optimize the prompting strategy, add support for other models, or improve the UI.
📜 License

Distributed under the MIT License. See LICENSE for more information.
