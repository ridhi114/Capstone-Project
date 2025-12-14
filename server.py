# server.py
from __future__ import annotations

import json
import os
import pickle
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ============================================================
#  FastAPI app
# ============================================================
app = FastAPI(title="RaceBrain – Bahrain")

# ============================================================
#  Bahrain pace & degradation + 2025-ish grid
# ============================================================

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

BASE_PACE: Dict[str, float] = {
    "SOFT": 92.3,
    "MEDIUM": 93.2,
    "HARD": 94.1,
}

DEGRADATION: Dict[str, float] = {
    "SOFT": 0.120,
    "MEDIUM": 0.085,
    "HARD": 0.060,
}

# (driver, team)
GRID_DRIVERS: List[tuple[str, str]] = [
    ("Lando Norris", "McLaren"),
    ("Oscar Piastri", "McLaren"),
    ("Max Verstappen", "Red Bull"),
    ("George Russell", "Mercedes"),
    ("Charles Leclerc", "Ferrari"),
    ("Lewis Hamilton", "Ferrari"),
    ("Kimi Antonelli", "Mercedes"),
    ("Alex Albon", "Williams"),
    ("Nico Hulkenberg", "Haas"),
    ("Esteban Ocon", "Alpine"),
    ("Sergio Perez", "Red Bull"),
    ("Carlos Sainz", "Ferrari"),
    ("Logan Sargeant", "Williams"),
    ("Pierre Gasly", "Alpine"),
    ("Fernando Alonso", "Aston Martin"),
    ("Lance Stroll", "Aston Martin"),
    ("Zhou Guanyu", "Sauber"),
    ("Valtteri Bottas", "Sauber"),
    ("Kevin Magnussen", "Haas"),
]

TEAMS: Dict[str, List[str]] = {}
for d, t in GRID_DRIVERS:
    TEAMS.setdefault(t, []).append(d)

DRIVER_TO_TEAM: Dict[str, str] = {d: t for d, t in GRID_DRIVERS}

# ============================================================
#  RL policy loading (from train_bahrain_policy.py)
# ============================================================

RL_POLICY_PATH = os.path.join("models", "rl_policy.pkl")
RL_TABLE: Optional[Dict[tuple, float]] = None
RL_BINS: Dict[str, int] = {"lap": 5, "age": 5}

COMP_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
IDX_TO_COMP = {v: k for k, v in COMP_TO_IDX.items()}


def load_rl_policy() -> None:
    """Load RL Q-table and bins from disk if available."""
    global RL_TABLE, RL_BINS
    if not os.path.exists(RL_POLICY_PATH):
        print("[BOOT] RL policy not found; using heuristic AI.")
        RL_TABLE = None
        return

    try:
        with open(RL_POLICY_PATH, "rb") as f:
            obj = pickle.load(f)
        RL_TABLE = obj.get("q_table", None)
        RL_BINS = obj.get("bins", {"lap": 5, "age": 5})
        print("[BOOT] RL policy loaded from models/rl_policy.pkl")
    except Exception as e:
        print("[BOOT] Failed to load RL policy:", e)
        RL_TABLE = None


load_rl_policy()

# ============================================================
#  Lap-time + simulation helpers
# ============================================================


def lap_mean_time(compound: str, tyre_age: int, driver_offset: float = 0.0) -> float:
    """Deterministic mean lap time for a given compound & age."""
    base = BASE_PACE[compound]
    slope = DEGRADATION[compound]
    return base + slope * max(1, tyre_age) + driver_offset


def simulate_driver(
    laps_total: int,
    pit_laps: List[int],
    seq: List[str],
    pit_loss: float,
    sigma: float,
    driver_offset: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[List[float], float]:
    """
    Simulate one driver:
    - seq: list of compounds in order
    - pit_laps: laps to pit (switch to next compound in seq)
    - pit_loss: added time on pit laps
    - sigma: lap-time noise
    """
    if not seq:
        raise ValueError("Tyre sequence cannot be empty.")

    pit_set = set(pit_laps)
    laps: List[float] = []

    if seed is not None:
        rnd = random.Random(seed)
        gauss = rnd.gauss
    else:
        gauss = random.gauss

    stint_idx = 0
    tyre_age = 0

    for lap in range(1, laps_total + 1):
        if stint_idx >= len(seq):
            stint_idx = len(seq) - 1

        comp = seq[stint_idx]
        if comp not in COMPOUNDS:
            raise ValueError(f"Unknown compound: {comp}")

        tyre_age += 1
        mu = lap_mean_time(comp, tyre_age, driver_offset)
        t = gauss(mu, sigma)

        if lap in pit_set:
            t += pit_loss
            stint_idx += 1
            tyre_age = 0

        laps.append(t)

    return laps, float(sum(laps))


# ============================================================
#  AI plan: RL → heuristic
# ============================================================


def _bin_value(v: int, step: int) -> int:
    return (int(v) // step) * step


def ai_plan_from_rl(
    laps_total: int,
    start_comp: str,
) -> Optional[Dict[str, Any]]:
    """
    Build a simple 1-stop plan from the learned Q-table (if present).
    State bins: lap, age, comp_idx; action: 0=stay,1=SOFT,2=MEDIUM,3=HARD.
    """
    if RL_TABLE is None:
        return None
    if start_comp not in COMP_TO_IDX:
        start_comp = "SOFT"

    lap_step = RL_BINS.get("lap", 5)
    age_step = RL_BINS.get("age", 5)

    lap = 1
    age = 1
    comp = start_comp
    pitted = False

    pits: List[int] = []
    seq: List[str] = [start_comp]

    while lap <= laps_total:
        lap_b = _bin_value(lap, lap_step)
        age_b = _bin_value(age, age_step)
        c_idx = COMP_TO_IDX[comp]

        best_a = 0
        best_q = -1e18
        for a in range(0, 4):  # 0=stay,1=SOFT,2=MED,3=HARD
            key = (lap_b, age_b, c_idx, a)
            q = RL_TABLE.get(key, -1e9)
            if q > best_q:
                best_q = q
                best_a = a

        # if already pitted or best action is "stay", just move on
        if pitted or best_a == 0:
            age += 1
            lap += 1
            continue

        # otherwise, we decide to pit now and switch compound
        new_comp = IDX_TO_COMP.get(best_a, comp)
        pits.append(lap)
        seq.append(new_comp)
        comp = new_comp
        age = 1
        pitted = True
        lap += 1

    if not pits:
        return None

    return {"pits": sorted(pits), "seq": seq, "source": "RL policy"}


def ai_plan_heuristic(
    laps_total: int,
    start_comp: str,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Simple 1-stop heuristic, tuned by difficulty.
    """
    if start_comp not in COMPOUNDS:
        start_comp = "MEDIUM"

    if difficulty == "easy":
        pit = int(laps_total * 0.40)
        next_comp = "HARD"
    elif difficulty == "hard":
        pit = int(laps_total * 0.55)
        next_comp = "HARD"
    else:  # normal
        pit = int(laps_total * 0.50)
        next_comp = "MEDIUM"

    pit = max(6, min(laps_total - 6, pit))
    return {
        "pits": [pit],
        "seq": [start_comp, next_comp],
        "source": "Heuristic",
    }


def build_ai_plan(
    laps_total: int,
    start_comp: str,
    difficulty: str,
) -> Dict[str, Any]:
    """
    Prefer RL plan; fallback to heuristic.
    """
    rl_plan = ai_plan_from_rl(laps_total, start_comp)
    if rl_plan is not None:
        return rl_plan
    return ai_plan_heuristic(laps_total, start_comp, difficulty)


# ============================================================
#  Request / response models
# ============================================================

class RaceRequest(BaseModel):
    team: str
    driver: str
    pit_laps: str
    seq: List[str]
    laps_total: int
    pit_loss: float
    sigma: float
    ai_start: str
    ai_diff: str  # "easy", "normal", "hard"


class RaceResponse(BaseModel):
    winner: str
    margin: float
    user_total: float
    ai_total: float
    laps: List[int]
    user_laps: List[float]
    ai_laps: List[float]
    gap: List[float]
    ai_plan: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    driver_stats: Dict[str, Any]
    info: Dict[str, Any]


# ============================================================
#  Per-driver stats + leaderboard (in memory)
# ============================================================

# key: "Driver (Team)" → stats
DRIVER_STATS: Dict[str, Dict[str, Any]] = {}


def _driver_key(driver: str, team: str) -> str:
    return f"{driver} ({team})"


def update_driver_stats(
    driver: str,
    team: str,
    winner: str,
    margin: float,
    user_total: float,
    ai_total: float,
) -> Dict[str, Any]:
    """
    Tracks per-driver stats:
      - runs, user_wins, ai_wins
      - avg_margin (seconds)
      - best_user_time, best_ai_time
    """
    k = _driver_key(driver, team)
    row = DRIVER_STATS.get(k)
    if row is None:
        row = {
            "driver": driver,
            "team": team,
            "runs": 0,
            "user_wins": 0,
            "ai_wins": 0,
            "avg_margin": 0.0,
            "best_user_time": None,
            "best_ai_time": None,
        }
        DRIVER_STATS[k] = row

    row["runs"] += 1
    if winner == "User":
        row["user_wins"] += 1
    else:
        row["ai_wins"] += 1

    # running average of margin (always positive)
    prev_avg = float(row["avg_margin"])
    n = row["runs"]
    row["avg_margin"] = (prev_avg * (n - 1) + float(margin)) / n

    if row["best_user_time"] is None or user_total < row["best_user_time"]:
        row["best_user_time"] = float(user_total)
    if row["best_ai_time"] is None or ai_total < row["best_ai_time"]:
        row["best_ai_time"] = float(ai_total)

    return row


def sorted_leaderboard() -> List[Dict[str, Any]]:
    """
    Convert DRIVER_STATS to the table shape the frontend expects.
    """
    rows = list(DRIVER_STATS.values())
    rows.sort(
        key=lambda r: (
            -r["user_wins"],  # more user wins at top
            r["avg_margin"],  # larger avg margin next
            r["best_user_time"] if r["best_user_time"] is not None else 9999.0,
        )
    )

    table: List[Dict[str, Any]] = []
    for r in rows:
        table.append(
            {
                "driver": r["driver"],
                "team": r["team"],
                "runs": r["runs"],
                "user_wins": r["user_wins"],
                "avg_margin": round(float(r["avg_margin"]), 3),
                "best_user": round(float(r["best_user_time"]), 3)
                if r["best_user_time"] is not None
                else None,
                "best_ai": round(float(r["best_ai_time"]), 3)
                if r["best_ai_time"] is not None
                else None,
            }
        )
    return table


def driver_stats_public(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shape of driver_stats field for the info box.
    """
    return {
        "driver": row["driver"],
        "team": row["team"],
        "runs": row["runs"],
        "user_wins": row["user_wins"],
        "ai_wins": row["ai_wins"],
        "avg_margin": round(float(row["avg_margin"]), 3),
        "best_user_time": round(float(row["best_user_time"]), 3)
        if row["best_user_time"] is not None
        else None,
        "best_ai_time": round(float(row["best_ai_time"]), 3)
        if row["best_ai_time"] is not None
        else None,
    }


# ============================================================
#  /race endpoint
# ============================================================

@app.post("/race", response_model=RaceResponse)
def race(req: RaceRequest) -> RaceResponse:
    try:
        # 1) parse pit laps
        pit_laps: List[int]
        if req.pit_laps.strip():
            try:
                pit_laps = sorted(
                    {
                        int(x.strip())
                        for x in req.pit_laps.split(",")
                        if x.strip()
                    }
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Pit laps must be comma-separated integers.",
                )
        else:
            pit_laps = []

        # 2) validate tyre sequence
        if len(req.seq) != len(pit_laps) + 1:
            raise HTTPException(
                status_code=400,
                detail="Tyre sequence must be exactly one longer than pit laps.",
            )

        if any(c not in COMPOUNDS for c in req.seq):
            raise HTTPException(
                status_code=400,
                detail="Unknown compound in tyre sequence.",
            )

        # 3) physics params
        laps_total = max(1, int(req.laps_total))
        pit_loss = float(req.pit_loss)
        sigma = max(0.0, float(req.sigma))

        # 4) user simulation
        user_laps, user_total = simulate_driver(
            laps_total=laps_total,
            pit_laps=pit_laps,
            seq=req.seq,
            pit_loss=pit_loss,
            sigma=sigma,
            driver_offset=0.0,
        )

        # 5) AI difficulty → offset & text (so AI can actually win on hard)
        diff = (req.ai_diff or "").lower()
        if diff == "easy":
            ai_offset = 1.0
            diff_desc = "Easy: AI is ~1.0s per lap slower on average."
        elif diff == "hard":
            ai_offset = -0.2
            diff_desc = "Hard: AI is slightly faster on average; you need a strong strategy to beat it."
        else:
            diff = "normal"
            ai_offset = 0.4
            diff_desc = "Normal: AI is close to your pace with a small disadvantage."

        # 6) AI plan
        ai_plan = build_ai_plan(
            laps_total=laps_total,
            start_comp=req.ai_start,
            difficulty=diff,
        )
        ai_pits = ai_plan["pits"]
        ai_seq = ai_plan["seq"]
        ai_source = ai_plan["source"]

        ai_laps, ai_total = simulate_driver(
            laps_total=laps_total,
            pit_laps=ai_pits,
            seq=ai_seq,
            pit_loss=pit_loss,
            sigma=sigma,
            driver_offset=ai_offset,
        )

        # 7) winner, margin
        if user_total < ai_total:
            winner = "User"
            margin = ai_total - user_total
        else:
            winner = "AI"
            margin = user_total - ai_total

        # 8) gap per lap (user - AI)
        laps = list(range(1, laps_total + 1))
        gap = [u - a for u, a in zip(user_laps, ai_laps)]

        # 9) update per-driver stats & leaderboard
        stats_row = update_driver_stats(
            driver=req.driver,
            team=req.team,
            winner=winner,
            margin=float(margin),
            user_total=float(user_total),
            ai_total=float(ai_total),
        )

        table_rows = sorted_leaderboard()
        stats_public = driver_stats_public(stats_row)

        # 10) info block for the AI info box
        info_obj = {
            "ai_source": ai_source,
            "note": f"{diff_desc} Start compound: {req.ai_start}. Strategy source: {ai_source}.",
        }

        return RaceResponse(
            winner=winner,
            margin=round(float(margin), 3),
            user_total=round(float(user_total), 3),
            ai_total=round(float(ai_total), 3),
            laps=laps,
            user_laps=[round(float(x), 3) for x in user_laps],
            ai_laps=[round(float(x), 3) for x in ai_laps],
            gap=[round(float(x), 3) for x in gap],
            ai_plan={
                "pits": ai_pits,
                "seq": ai_seq,
                "source": ai_source,
            },
            leaderboard=table_rows,
            driver_stats=stats_public,
            info=info_obj,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("[/race] Internal error:", repr(e))
        raise HTTPException(status_code=500, detail="Internal simulation error: " + str(e))


# ============================================================
#  Frontend HTML (F1 red theme, charts, leaderboard + driver box)
# ============================================================

# ---------------- FRONTEND TEMPLATES ----------------

TEAM_DRIVERS_JSON = json.dumps(TEAMS)

# Part 1: START_TEMPLATE
START_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RaceBrain - Start</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: #2a2a2a;
            border-radius: 12px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 {
            color: #e10600;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: #999;
            margin-bottom: 40px;
            font-size: 0.95em;
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #ddd;
            font-weight: 500;
        }
        select {
            width: 100%;
            padding: 12px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            color: #fff;
            font-size: 1em;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        select:hover { border-color: #e10600; }
        select:focus {
            outline: none;
            border-color: #e10600;
        }
        .preset-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .preset-btn {
            padding: 12px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9em;
        }
        .preset-btn:hover {
            border-color: #e10600;
            background: #2a2a2a;
        }
        .start-btn {
            width: 100%;
            padding: 16px;
            background: #e10600;
            border: none;
            border-radius: 6px;
            color: #fff;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 30px;
        }
        .start-btn:hover { background: #c00500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏎️ RaceBrain</h1>
        <p class="subtitle">Formula 1 Race Strategy Simulator</p>
        
        <form id="startForm">
            <div class="form-group">
                <label>Mode</label>
                <select id="mode" name="mode">
                    <option value="user_vs_ai">User vs AI</option>
                    <option value="ai_vs_ai">AI vs AI</option>
                    <option value="practice">Practice</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Track</label>
                <select id="track" name="track">
                    <option value="Bahrain">Bahrain</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Difficulty</label>
                <select id="difficulty" name="difficulty">
                    <option value="easy">Easy</option>
                    <option value="normal" selected>Normal</option>
                    <option value="hard">Hard</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Team</label>
                <select id="team" name="team">
                    <option value="McLaren">McLaren</option>
                    <option value="Red Bull">Red Bull</option>
                    <option value="Ferrari">Ferrari</option>
                    <option value="Mercedes">Mercedes</option>
                    <option value="Aston Martin">Aston Martin</option>
                    <option value="Alpine">Alpine</option>
                    <option value="Williams">Williams</option>
                    <option value="RB">RB</option>
                    <option value="Haas">Haas</option>
                    <option value="Sauber">Sauber</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Driver</label>
                <select id="driver" name="driver">
                    <option value="Piastri">Piastri</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Quick Start Presets</label>
                <div class="preset-grid">
                    <button type="button" class="preset-btn" onclick="applyPreset('mclaren')">McLaren / Piastri</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('redbull')">Red Bull / Verstappen</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('ferrari')">Ferrari / Leclerc</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('mercedes')">Mercedes / Hamilton</button>
                </div>
            </div>
            
            <button type="submit" class="start-btn">Start Race →</button>
        </form>
    </div>
    
    <script>
        const teamDrivers = {
            "McLaren": ["Norris", "Piastri"],
            "Red Bull": ["Verstappen", "Perez"],
            "Ferrari": ["Leclerc", "Sainz"],
            "Mercedes": ["Hamilton", "Russell"],
            "Aston Martin": ["Alonso", "Stroll"],
            "Alpine": ["Gasly", "Ocon"],
            "Williams": ["Albon", "Colapinto"],
            "RB": ["Tsunoda", "Lawson"],
            "Haas": ["Hulkenberg", "Bearman"],
            "Sauber": ["Bottas", "Zhou"]
        };
        
        const teamSelect = document.getElementById('team');
        const driverSelect = document.getElementById('driver');
        
        teamSelect.addEventListener('change', function() {
            const drivers = teamDrivers[this.value] || [];
            driverSelect.innerHTML = drivers.map(function(d) { return '<option value="' + d + '">' + d + '</option>'; }).join('');
        });
        
        function applyPreset(preset) {
            const presets = {
                'mclaren': { team: 'McLaren', driver: 'Piastri' },
                'redbull': { team: 'Red Bull', driver: 'Verstappen' },
                'ferrari': { team: 'Ferrari', driver: 'Leclerc' },
                'mercedes': { team: 'Mercedes', driver: 'Hamilton' }
            };
            const p = presets[preset];
            teamSelect.value = p.team;
            teamSelect.dispatchEvent(new Event('change'));
            setTimeout(function() { driverSelect.value = p.driver; }, 50);
        }
        
        document.getElementById('startForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const params = new URLSearchParams({
                mode: document.getElementById('mode').value,
                track: document.getElementById('track').value,
                difficulty: document.getElementById('difficulty').value,
                team: document.getElementById('team').value,
                driver: document.getElementById('driver').value
            });
            window.location.href = '/game?' + params.toString();
        });
    </script>
</body>
</html>
"""

# Part 2: INDEX_TEMPLATE  
INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RaceBrain — Bahrain GP Strategy Simulator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Formula1:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --f1-red: #E10600;
            --f1-dark: #15151E;
            --f1-gray: #38383F;
            --f1-light-gray: #F7F4F1;
            --soft-red: #FF1E1E;
            --medium-yellow: #FFD700;
            --hard-white: #E8E8E8;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #15151E 0%, #1E1E2E 100%);
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        .bg-animated {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(225, 6, 0, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(225, 6, 0, 0.08) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Header */
        .header {
            background: linear-gradient(90deg, var(--f1-red) 0%, #C00500 100%);
            padding: 24px 40px;
            box-shadow: 0 4px 24px rgba(225, 6, 0, 0.3);
            position: relative;
            z-index: 10;
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: #fff;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }
        
        .logo-text h1 {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .logo-text p {
            font-size: 13px;
            opacity: 0.9;
            margin-top: 2px;
        }
        
        .track-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        
        /* Winner Banner */
        .winner-banner {
            background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
            color: var(--f1-dark);
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            display: none;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 24px rgba(255, 215, 0, 0.4);
        }
        
        .winner-banner.show { 
            display: block; 
            animation: slideDown 0.5s ease-out;
        }
        
        @keyframes slideDown {
            from { transform: translateY(-100%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .confetti {
            position: absolute;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: confettiFall 2s ease-out forwards;
        }
        
        @keyframes confettiFall {
            0% { top: -20px; opacity: 1; }
            100% { top: 110%; opacity: 0; transform: translateY(20px) rotate(720deg); }
        }
        
        /* Main Container */
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 20px;
            position: relative;
            z-index: 1;
        }
        
        /* Control Panel */
        .control-panel {
            background: rgba(56, 56, 63, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--f1-red);
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .section-title::before {
            content: '';
            width: 4px;
            height: 24px;
            background: var(--f1-red);
            border-radius: 2px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        select, input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(21, 21, 30, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #fff;
            font-size: 15px;
            transition: all 0.3s;
        }
        
        select:hover, input:hover {
            border-color: var(--f1-red);
            background: rgba(21, 21, 30, 0.95);
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: var(--f1-red);
            box-shadow: 0 0 0 3px rgba(225, 6, 0, 0.2);
        }
        
        select[multiple] {
            height: 120px;
            padding: 8px;
        }
        
        /* Preset Buttons */
        .preset-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 12px;
        }
        
        .preset-btn {
            padding: 10px 16px;
            background: rgba(225, 6, 0, 0.1);
            border: 1px solid rgba(225, 6, 0, 0.3);
            border-radius: 8px;
            color: #fff;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .preset-btn:hover {
            background: var(--f1-red);
            border-color: var(--f1-red);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(225, 6, 0, 0.4);
        }
        
        /* Action Buttons */
        .action-row {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }
        
        button {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--f1-red) 0%, #C00500 100%);
            color: #fff;
            box-shadow: 0 4px 16px rgba(225, 6, 0, 0.4);
            flex: 1;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(225, 6, 0, 0.6);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Info Box */
        .info-box {
            background: rgba(225, 6, 0, 0.1);
            border-left: 3px solid var(--f1-red);
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13px;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 12px;
        }
        
        /* Strategy Visualizer */
        .strategy-panel {
            background: rgba(56, 56, 63, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .strategy-row {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
        }
        
        .strategy-label {
            width: 80px;
            font-weight: 700;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .strategy-bar {
            flex: 1;
            display: flex;
            height: 48px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .stint-block {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 12px;
            font-weight: 700;
            border-right: 2px solid rgba(0, 0, 0, 0.2);
            transition: all 0.3s;
        }
        
        .stint-block:hover {
            filter: brightness(1.1);
            transform: scaleY(1.05);
        }
        
        .stint-block:last-child { border-right: none; }
        .stint-soft { background: linear-gradient(135deg, #FF1E1E 0%, #E10600 100%); }
        .stint-medium { background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: #1a1a1a; }
        .stint-hard { background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%); color: #1a1a1a; }
        
        /* Charts */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }
        
        .chart-card {
            background: rgba(56, 56, 63, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .chart-title {
            font-size: 16px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 20px;
        }
        
        canvas {
            max-height: 300px;
        }
        
        /* Leaderboard */
        .leaderboard {
            background: rgba(56, 56, 63, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        
        th {
            background: rgba(225, 6, 0, 0.1);
            padding: 12px;
            text-align: left;
            font-weight: 700;
            color: var(--f1-red);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 11px;
            border-bottom: 2px solid var(--f1-red);
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.9);
        }
        
        tr:hover {
            background: rgba(225, 6, 0, 0.05);
        }
        
        /* Tooltip */
        .info-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(225, 6, 0, 0.2);
            color: var(--f1-red);
            font-size: 11px;
            font-weight: 700;
            cursor: help;
            margin-left: 6px;
        }
        
        #tooltip {
            position: fixed;
            background: rgba(21, 21, 30, 0.95);
            backdrop-filter: blur(10px);
            color: #fff;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 280px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            border: 1px solid rgba(225, 6, 0, 0.3);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        }
        
        /* Responsive */
        @media (max-width: 900px) {
            .grid, .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .preset-grid {
                grid-template-columns: 1fr;
            }
            
            .action-row {
                flex-direction: column;
            }
            
            .header-content {
                flex-direction: column;
                gap: 16px;
            }
        }
        
        /* Loading Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="bg-animated"></div>
    
    <!-- Header -->
    <div class="header">
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🏎️</div>
                <div class="logo-text">
                    <h1>RaceBrain</h1>
                    <p>Formula 1 Strategy Simulator</p>
                </div>
            </div>
            <div class="track-badge">🇧🇭 Bahrain International Circuit</div>
        </div>
    </div>
    
    <!-- Winner Banner -->
    <div id="winnerBanner" class="winner-banner"></div>
    
    <!-- Main Content -->
    <main>
        <!-- Control Panel -->
        <div class="control-panel">
            <div class="section-title">Race Configuration</div>
            
            <div class="grid">
                <!-- Your Car -->
                <div>
                    <h3 style="font-size: 18px; font-weight: 700; color: rgba(255,255,255,0.9); margin-bottom: 20px;">🏁 Your Car</h3>
                    
                    <div class="input-group">
                        <label>Team</label>
                        <select id="teamSelect"></select>
                    </div>
                    
                    <div class="input-group">
                        <label>Driver</label>
                        <select id="driverSelect"></select>
                    </div>
                    
                    <div class="input-group">
                        <label>Pit Laps <span class="info-icon" data-info="Enter lap numbers separated by commas. Example: 18,38 for pits on laps 18 and 38">i</span></label>
                        <input id="pitLaps" type="text" placeholder="e.g., 18,38" />
                    </div>
                    
                    <div class="input-group">
                        <label>Tyre Strategy <span class="info-icon" data-info="Hold Ctrl/Cmd to select multiple. Must have exactly one more compound than pit stops">i</span></label>
                        <select id="tyreSeq" multiple>
                            <option value="SOFT">🔴 SOFT</option>
                            <option value="MEDIUM">🟡 MEDIUM</option>
                            <option value="HARD">⚪ HARD</option>
                        </select>
                    </div>
                    
                    <div style="margin-top: 16px;">
                        <label style="margin-bottom: 12px;">Quick Presets</label>
                        <div class="preset-grid">
                            <button class="preset-btn" onclick="applyPreset('classic')">Classic 2-Stop</button>
                            <button class="preset-btn" onclick="applyPreset('hard')">Hard Start</button>
                            <button class="preset-btn" onclick="applyPreset('sprint')">Sprint S→M</button>
                        </div>
                    </div>
                </div>
                
                <!-- AI Configuration -->
                <div>
                    <h3 style="font-size: 18px; font-weight: 700; color: rgba(255,255,255,0.9); margin-bottom: 20px;">🤖 AI Opponent</h3>
                    
                    <div class="input-group">
                        <label>Starting Compound</label>
                        <select id="aiStart">
                            <option value="SOFT">🔴 SOFT</option>
                            <option value="MEDIUM" selected>🟡 MEDIUM</option>
                            <option value="HARD">⚪ HARD</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Difficulty <span class="info-icon" data-info="Easy: AI is ~1s slower. Normal: Close racing. Hard: AI can be faster">i</span></label>
                        <select id="aiDiff">
                            <option value="easy">Easy</option>
                            <option value="normal" selected>Normal</option>
                            <option value="hard">Hard</option>
                        </select>
                    </div>
                    
                    <div class="info-box" id="aiInfoBox">
                        Normal: AI is close to your pace with a small disadvantage.
                    </div>
                    
                    <div style="margin-top: 24px; padding: 16px; background: rgba(225,6,0,0.05); border-radius: 8px; border: 1px solid rgba(225,6,0,0.2);">
                        <div style="font-size: 12px; font-weight: 600; color: rgba(255,255,255,0.6); margin-bottom: 8px;">RACE PARAMETERS</div>
                        <div class="input-group" style="margin-bottom: 12px;">
                            <label style="font-size: 11px;">Total Laps</label>
                            <input id="lapsTotal" type="number" value="57" style="padding: 8px 12px;" />
                        </div>
                        <div class="input-group" style="margin-bottom: 12px;">
                            <label style="font-size: 11px;">Pit Loss (seconds)</label>
                            <input id="pitLoss" type="number" step="0.1" value="22.5" style="padding: 8px 12px;" />
                        </div>
                        <div class="input-group" style="margin-bottom: 0;">
                            <label style="font-size: 11px;">Lap Time Variance</label>
                            <input id="sigma" type="number" step="0.05" value="0.35" style="padding: 8px 12px;" />
                        </div>
                        <div style="margin-top: 12px; font-size: 11px; color: rgba(255,255,255,0.5);">
                            Base pace: SOFT 92.3s | MEDIUM 93.2s | HARD 94.1s
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Action Buttons -->
            <div class="action-row">
                <button class="btn-primary" onclick="startRace()">🏁 Start Race</button>
                <button class="btn-secondary" onclick="resetCharts()">Reset Charts</button>
                <button class="btn-secondary" onclick="resetLeaderboard()">Reset Stats</button>
            </div>
            
            <div class="info-box" id="summaryBox">
                Select your driver and strategy, then click Start Race to begin the simulation.
            </div>
        </div>
        
        <!-- Strategy Visualizer -->
        <div class="strategy-panel">
            <div class="section-title">Strategy Timeline</div>
            <div class="strategy-row">
                <div class="strategy-label">YOU</div>
                <div class="strategy-bar" id="userStrategyBar">
                    <div style="flex: 1; background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.4); font-size: 13px;">
                        Run a race to see your strategy
                    </div>
                </div>
            </div>
            <div class="strategy-row">
                <div class="strategy-label">AI</div>
                <div class="strategy-bar" id="aiStrategyBar">
                    <div style="flex: 1; background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.4); font-size: 13px;">
                        Run a race to see AI strategy
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-title">📊 Lap-by-Lap Performance</div>
                <canvas id="lapChart"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">⏱️ Time Gap Analysis</div>
                <canvas id="gapChart"></canvas>
            </div>
        </div>
        
        <!-- Leaderboard -->
        <div class="leaderboard">
            <div class="section-title">Driver Statistics</div>
            <div id="driverStatsBox" style="margin-bottom: 20px; padding: 16px; background: rgba(225,6,0,0.1); border-radius: 8px; font-size: 13px; color: rgba(255,255,255,0.9);">
                No stats yet. Run your first race!
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Driver</th>
                        <th>Team</th>
                        <th>Races</th>
                        <th>Wins</th>
                        <th>Avg Margin</th>
                        <th>Best User</th>
                        <th>Best AI</th>
                    </tr>
                </thead>
                <tbody id="leaderboardBody">
                    <tr>
                        <td colspan="8" style="text-align: center; color: rgba(255,255,255,0.4); padding: 24px;">
                            No races completed yet. Start racing to build your statistics!
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </main>
    
    
    <script>
        const TEAM_DRIVERS = __TEAM_DRIVERS__;
        
        let lapChart = null;
        let gapChart = null;
        
        function initTeams() {
            const params = new URLSearchParams(window.location.search);
            const qTeam = params.get('team');
            const qDriver = params.get('driver');
            const qDifficulty = params.get('difficulty');
            
            const teamSelect = document.getElementById('teamSelect');
            Object.keys(TEAM_DRIVERS).sort().forEach(function(team) {
                const opt = document.createElement('option');
                opt.value = team;
                opt.textContent = team;
                teamSelect.appendChild(opt);
            });
            
            if (qTeam && TEAM_DRIVERS[qTeam]) {
                teamSelect.value = qTeam;
            }
            
            refreshDrivers();
            
            if (qDriver) {
                const driverSelect = document.getElementById('driverSelect');
                const options = Array.from(driverSelect.options);
                const idx = options.findIndex(function(o) { return o.value === qDriver; });
                if (idx >= 0) driverSelect.selectedIndex = idx;
            }
            
            if (qDifficulty) {
                document.getElementById('aiDiff').value = qDifficulty;
            }
            
            teamSelect.addEventListener('change', refreshDrivers);
            document.getElementById('aiDiff').addEventListener('change', updateAiInfo);
            updateAiInfo();
        }
        
        function refreshDrivers() {
            const team = document.getElementById('teamSelect').value;
            const driverSelect = document.getElementById('driverSelect');
            driverSelect.innerHTML = '';
            (TEAM_DRIVERS[team] || []).forEach(function(d) {
                const opt = document.createElement('option');
                opt.value = d;
                opt.textContent = d;
                driverSelect.appendChild(opt);
            });
        }
        
        function applyPreset(name) {
            const pitLaps = document.getElementById('pitLaps');
            const tyreSeq = document.getElementById('tyreSeq');
            
            if (name === 'classic') {
                pitLaps.value = '18,38';
                selectTyres(['SOFT', 'MEDIUM', 'HARD']);
            } else if (name === 'hard') {
                pitLaps.value = '30';
                selectTyres(['HARD', 'MEDIUM']);
            } else if (name === 'sprint') {
                pitLaps.value = '20';
                selectTyres(['SOFT', 'MEDIUM']);
            }
        }
        
        function selectTyres(list) {
            const sel = document.getElementById('tyreSeq');
            Array.from(sel.options).forEach(function(o) {
                o.selected = list.indexOf(o.value) >= 0;
            });
        }
        
        function updateAiInfo() {
            const diff = document.getElementById('aiDiff').value;
            const box = document.getElementById('aiInfoBox');
            if (diff === 'easy') {
                box.textContent = 'Easy: AI is ~1.0s per lap slower on average.';
            } else if (diff === 'hard') {
                box.textContent = 'Hard: AI is slightly faster on average; you need a strong strategy to beat it.';
            } else {
                box.textContent = 'Normal: AI is close to your pace with a small disadvantage.';
            }
        }
        
        async function startRace() {
            const seq = Array.from(document.getElementById('tyreSeq').selectedOptions).map(function(o) { return o.value; });
            if (seq.length === 0) {
                alert('Select at least one compound in tyre sequence');
                return;
            }
            
            const payload = {
                team: document.getElementById('teamSelect').value,
                driver: document.getElementById('driverSelect').value,
                pit_laps: document.getElementById('pitLaps').value,
                seq: seq,
                laps_total: parseInt(document.getElementById('lapsTotal').value),
                pit_loss: parseFloat(document.getElementById('pitLoss').value),
                sigma: parseFloat(document.getElementById('sigma').value),
                ai_start: document.getElementById('aiStart').value,
                ai_diff: document.getElementById('aiDiff').value
            };
            
            try {
                const res = await fetch('/race', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) {
                    alert('Error: ' + (data.detail || 'Unknown error'));
                    return;
                }
                handleRaceResult(data);
            } catch (err) {
                alert('Could not reach backend: ' + err.message);
            }
        }
        
        function fireConfetti() {
            const banner = document.getElementById('winnerBanner');
            const colors = ['#fff', '#ffd700', '#ff6b6b', '#4ecdc4', '#1a9cff'];
            for (let i = 0; i < 60; i++) {
                const piece = document.createElement('span');
                piece.className = 'confetti';
                piece.style.left = Math.random() * 100 + '%';
                piece.style.background = colors[Math.floor(Math.random() * colors.length)];
                piece.style.animationDelay = (Math.random() * 0.4) + 's';
                banner.appendChild(piece);
                setTimeout(function() { piece.remove(); }, 2000);
            }
        }
        
        function buildStrategyBar(totalLaps, pits, seq) {
            const stints = [];
            let prevLap = 0;
            
            pits.forEach(function(pitLap, idx) {
                const laps = pitLap - prevLap;
                stints.push({ compound: seq[idx], start: prevLap + 1, end: pitLap, laps: laps });
                prevLap = pitLap;
            });
            
            const lastLaps = totalLaps - prevLap;
            stints.push({ compound: seq[seq.length - 1], start: prevLap + 1, end: totalLaps, laps: lastLaps });
            
            return stints.map(function(stint) {
                const className = 'stint-' + stint.compound.toLowerCase();
                const label = stint.compound.charAt(0) + ' (' + stint.start + '–' + stint.end + ')';
                return '<div class="stint-block ' + className + '" style="flex: ' + stint.laps + '">' + label + '</div>';
            }).join('');
        }
        
        function handleRaceResult(data) {
            const banner = document.getElementById('winnerBanner');
            banner.textContent = 'Winner: ' + data.winner + ' — Margin: ' + data.margin.toFixed(3) + ' s';
            banner.className = 'winner-banner show';
            fireConfetti();
            
            const aiSeq = data.ai_plan.seq.join(' → ');
            const aiPits = data.ai_plan.pits.length ? (' @ pits ' + data.ai_plan.pits.join(', ')) : ' (no-stop)';
            document.getElementById('summaryBox').innerHTML =
                'AI plan (' + data.ai_plan.source + '): ' + aiSeq + aiPits + '. ' +
                'User total: <b>' + data.user_total.toFixed(3) + 's</b>, AI total: <b>' + data.ai_total.toFixed(3) + 's</b>. ' +
                'You were <b>' + data.margin.toFixed(3) + 's</b> from AI.';
            
            document.getElementById('aiInfoBox').textContent = data.info.note;
            
            const userPits = document.getElementById('pitLaps').value.split(',').map(function(x) { return parseInt(x.trim()); }).filter(function(x) { return !isNaN(x); });
            const userSeq = Array.from(document.getElementById('tyreSeq').selectedOptions).map(function(o) { return o.value; });
            const totalLaps = parseInt(document.getElementById('lapsTotal').value);
            
            document.getElementById('userStrategyBar').innerHTML = buildStrategyBar(totalLaps, userPits, userSeq);
            document.getElementById('aiStrategyBar').innerHTML = buildStrategyBar(totalLaps, data.ai_plan.pits, data.ai_plan.seq);
            
            updateCharts(data);
            updateLeaderboard(data.leaderboard);
            updateDriverStatsBox(data.driver_stats);
        }
        
        function updateCharts(data) {
            const ctx1 = document.getElementById('lapChart').getContext('2d');
            const ctx2 = document.getElementById('gapChart').getContext('2d');
            
            if (lapChart) lapChart.destroy();
            if (gapChart) gapChart.destroy();
            
            lapChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: data.laps,
                    datasets: [
                        { label: 'User', data: data.user_laps, borderColor: '#e10600', borderWidth: 2, fill: false },
                        { label: 'AI', data: data.ai_laps, borderColor: '#666', borderWidth: 2, borderDash: [4,2], fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true } },
                    scales: {
                        x: { title: { display: true, text: 'Lap' } },
                        y: { title: { display: true, text: 'Lap time (s)' } }
                    }
                }
            });
            
            gapChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: data.laps,
                    datasets: [
                        { label: 'User − AI gap (s)', data: data.gap, borderColor: '#4ecdc4', borderWidth: 2, fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { title: { display: true, text: 'Lap' } },
                        y: { title: { display: true, text: 'Δ time (s)' } }
                    }
                }
            });
        }
        
        function updateLeaderboard(rows) {
            const tbody = document.getElementById('leaderboardBody');
            tbody.innerHTML = '';
            if (rows.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #999;">No races completed yet</td></tr>';
                return;
            }
            rows.forEach(function(row, idx) {
                const tr = document.createElement('tr');
                const bestUser = row.best_user == null ? '—' : row.best_user;
                const bestAI = row.best_ai == null ? '—' : row.best_ai;
                tr.innerHTML = '<td>' + (idx + 1) + '</td>' +
                    '<td>' + row.driver + '</td>' +
                    '<td>' + row.team + '</td>' +
                    '<td>' + row.runs + '</td>' +
                    '<td>' + row.user_wins + '</td>' +
                    '<td>' + row.avg_margin + '</td>' +
                    '<td>' + bestUser + '</td>' +
                    '<td>' + bestAI + '</td>';
                tbody.appendChild(tr);
            });
        }
        
        function updateDriverStatsBox(stats) {
            const box = document.getElementById('driverStatsBox');
            if (!stats || !stats.runs) {
                box.textContent = 'No accumulated stats yet for this driver.';
                return;
            }
            const bu = stats.best_user_time ? stats.best_user_time.toFixed(3) + ' s' : '—';
            const ba = stats.best_ai_time ? stats.best_ai_time.toFixed(3) + ' s' : '—';
            box.innerHTML = '<b>' + stats.driver + ' (' + stats.team + ')</b>: ' +
                stats.user_wins + ' user win(s), ' + stats.ai_wins + ' AI win(s), ' +
                'avg margin: ' + stats.avg_margin.toFixed(3) + ' s, ' +
                'best user: ' + bu + ', best AI: ' + ba;
        }
        
        function resetCharts() {
            if (lapChart) lapChart.destroy();
            if (gapChart) gapChart.destroy();
            lapChart = null;
            gapChart = null;
            document.getElementById('summaryBox').innerHTML = 'Charts cleared. Run another race to see lap-by-lap data.';
            document.getElementById('winnerBanner').className = 'winner-banner';
            document.getElementById('userStrategyBar').innerHTML = '<div style="flex: 1; background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.4); font-size: 13px;">Run a race to see your strategy</div>';
            document.getElementById('aiStrategyBar').innerHTML = '<div style="flex: 1; background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; color: rgba(255,255,255,0.4); font-size: 13px;">Run a race to see AI strategy</div>';
        }
        
        function resetLeaderboard() {
            fetch('/reset_leaderboard', { method: 'POST' })
                .then(function() {
                    document.getElementById('leaderboardBody').innerHTML = '<tr><td colspan="8" style="text-align: center; color: rgba(255,255,255,0.4); padding: 24px;">No races completed yet</td></tr>';
                    document.getElementById('driverStatsBox').textContent = 'No stats yet. Run your first race!';
                })
                .catch(function(err) {
                    console.error('Failed to reset leaderboard:', err);
                });
        }
        
        // Tooltip handling
        document.addEventListener('mouseover', function(e) {
            if (e.target.classList.contains('info-icon')) {
                const tooltip = document.getElementById('tooltip');
                tooltip.textContent = e.target.getAttribute('data-info');
                tooltip.style.display = 'block';
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY + 10 + 'px';
            }
        });
        
        document.addEventListener('mouseout', function(e) {
            if (e.target.classList.contains('info-icon')) {
                document.getElementById('tooltip').style.display = 'none';
            }
        });
        
        // Initialize on page load
        window.addEventListener('DOMContentLoaded', function() {
            initTeams();
        });
    </script>
</body>
</html>
"""
START_HTML = START_TEMPLATE
INDEX_HTML = INDEX_TEMPLATE.replace("__TEAM_DRIVERS__", TEAM_DRIVERS_JSON)


@app.post("/reset_leaderboard")
def reset_leaderboard_endpoint():
    DRIVER_STATS.clear()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def start(_: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)  # Skip start page, go directly to game


@app.get("/game", response_class=HTMLResponse)
def game(_: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)