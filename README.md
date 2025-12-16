# 🏁 RaceBrain: An Interactive Formula 1 Strategy Simulator Using Reinforcement Learning

RaceBrain is an interactive web-based Formula 1 strategy simulator that allows users to race against an AI powered by reinforcement learning. The system is built using real Bahrain Grand Prix race data and integrates Reinforcement Learning, Monte Carlo simulation, and brute-force optimization to study pit-stop decision-making under uncertainty.

---

## 🚀 Features

- **Interactive User vs AI racing dashboard**
- **Realistic race simulation** using empirical tyre degradation models
- **AI opponent** powered by Q-learning
- **Strategy validation** using:
  - Deterministic brute-force optimization
  - Monte Carlo expected-value optimization
- **Dynamic lap-by-lap visualizations**
- **Session leaderboard** for repeatable experiments

---

## 🧠 Core Idea

Pit-stop strategy in Formula 1 is a **sequential decision problem under uncertainty**. RaceBrain demonstrates how reinforcement learning learns robust strategies that may differ from deterministic optimal solutions when race dynamics are stochastic.

---

## 🛠️ Tech Stack

### Backend
- **Python**
- **NumPy, Pandas** – data processing
- **Custom Q-learning implementation**
- **Monte Carlo simulation engine**
- **FastAPI** – API server & orchestration

### Frontend
- **HTML, CSS, JavaScript**
- **Chart.js** – interactive charts
- **REST-based communication** with backend

---

## 🧪 Methods Implemented

### 1. Reinforcement Learning (Q-Learning)
- **State:** lap number, tyre compound, tyre age
- **Action:** pit or continue
- **Reward:** negative total race time (terminal reward)
- **Trained under stochastic lap-time noise**

### 2. Brute-Force Optimization
- Exhaustive search over all pit-stop laps
- Deterministic race dynamics
- Provides a theoretical lower bound

### 3. Monte Carlo Simulation
- Expected-value optimization under uncertainty
- Multiple rollouts per pit-lap candidate

---

## 📊 Validation

Strategies learned by RL are compared against brute-force and Monte Carlo solutions. Observed divergence between RL and brute-force solutions highlights **risk-aware optimization**, not model error.

---

## 🎮 Game Dashboard

The RaceBrain dashboard allows users to:

- Select team and driver
- Choose tyre sequence and pit-stop strategy
- Configure race physics parameters
- Race against an AI using learned strategies
- Visualize lap times, gaps, and outcomes
- Track results using a session leaderboard

---

## ▶️ Running the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/racebrain.git
cd racebrain
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Backend Server
```bash
uvicorn server:app --reload
```

### 4. Open in Browser
Navigate to: `http://127.0.0.1:8000`

---

## 📁 Project Structure
```
racebrain/
│
├── server.py                           # FastAPI backend & frontend routing
├── train_bahrain_policy.py            # RL training script
├── data/
│   ├── bahrain_2025_raw.parquet
│   ├── bahrain_2025_clean.parquet
│   └── bahrain_2025_decay.pkl
│
├── leaderboard.json                    # Session leaderboard storage
├── notebooks/
│   └── Capstone_REAL_DATA_FINAL_FIX.ipynb
│
├── static/                             # Frontend assets
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## ⚠️ Limitations

- Tabular Q-learning (no neural networks)
- Single-stop strategies only
- No traffic, safety cars, or multi-agent interactions
- Sparse terminal reward structure

---

## 🔮 Future Work

- **Deep Reinforcement Learning** (DQN / Actor-Critic)
- **Multi-stop and adaptive strategies**
- **Traffic and safety car modeling**
- **Multi-agent race simulations**
- **Real-time strategy recommendation systems**

---

## 📚 References

- Stochastic Optimization & Robust Decision-Making: [DOI:10.1007/s00521-020-04871-1](https://doi.org/10.1007/s00521-020-04871-1)
- Reinforcement Learning for Sequential Decision Problems: [arXiv:2306.16088](https://arxiv.org/abs/2306.16088)
- Learning-Based Strategy Optimization under Uncertainty: [arXiv:2501.04068](https://arxiv.org/abs/2501.04068)

---

## 👩‍💻 Authors

**Capstone Project** — Bachelor of Data Science  
SP Jain School of Global Management

---

## 📜 License

This project is intended for **academic and educational use**.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/racebrain/issues).

---

## ⭐ Acknowledgments

Special thanks to the instructors and peers at SP Jain School of Global Management for their support throughout this capstone project.
