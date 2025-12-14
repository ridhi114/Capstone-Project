"""
Comprehensive RL Training for F1 Strategy
Trains Q-Learning model for ALL compound combinations
"""

import numpy as np
import pickle
import os
from collections import defaultdict

# Bahrain-specific constants
RACE_LAPS = 57
PIT_LOSS = 22.5

# Tyre data
BASE_PACE = {"SOFT": 92.3, "MEDIUM": 93.2, "HARD": 94.1}
DEGRADATION = {"SOFT": 0.120, "MEDIUM": 0.085, "HARD": 0.060}
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMP_TO_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
IDX_TO_COMP = {v: k for k, v in COMP_TO_IDX.items()}

# State binning
LAP_BIN = 5
AGE_BIN = 5


def lap_time(compound, tyre_age):
    """Calculate lap time based on compound and tyre age"""
    base = BASE_PACE[compound]
    deg = DEGRADATION[compound]
    return base + deg * max(1, tyre_age)


def bin_value(v, step):
    """Bin a value to the nearest step"""
    return (int(v) // step) * step


class F1StrategyEnv:
    """F1 Race Strategy Environment for Q-Learning"""
    
    def __init__(self, start_compound):
        self.start_compound = start_compound
        self.reset()
    
    def reset(self):
        """Reset environment to start of race"""
        self.lap = 1
        self.tyre_age = 1
        self.compound = self.start_compound
        self.pitted = False
        return self.get_state()
    
    def get_state(self):
        """Get binned state representation"""
        lap_b = bin_value(self.lap, LAP_BIN)
        age_b = bin_value(self.tyre_age, AGE_BIN)
        comp_idx = COMP_TO_IDX[self.compound]
        return (lap_b, age_b, comp_idx)
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        Actions: 0=stay, 1=pit_to_SOFT, 2=pit_to_MEDIUM, 3=pit_to_HARD
        """
        # Negative lap time as reward (we want to minimize)
        reward = -lap_time(self.compound, self.tyre_age)
        
        # Check if action is valid
        if action == 0 or self.pitted:
            # Stay on current tyres
            self.tyre_age += 1
        else:
            # Pit and change compound
            new_compound = IDX_TO_COMP[action]
            if new_compound != self.compound:  # Only pit if changing compound
                reward -= PIT_LOSS
                self.compound = new_compound
                self.tyre_age = 1
                self.pitted = True
            else:
                # Penalize trying to pit to same compound
                reward -= 100
                self.tyre_age += 1
        
        self.lap += 1
        done = self.lap > RACE_LAPS
        next_state = self.get_state()
        
        return next_state, reward, done


def train_qlearning(start_compound, episodes=5000, alpha=0.15, gamma=0.95):
    """
    Train Q-Learning agent for a specific starting compound
    """
    print(f"\nTraining for START={start_compound}")
    
    Q = defaultdict(float)
    env = F1StrategyEnv(start_compound)
    
    # Epsilon decay
    eps_start = 0.3
    eps_end = 0.02
    
    for ep in range(episodes):
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-ep / (episodes / 4))
        
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get valid actions (can't pit if already pitted, can't pit to same compound)
            if env.pitted:
                valid_actions = [0]
            else:
                valid_actions = [0]  # Can always stay
                # Can pit to different compounds
                for a in range(1, 4):
                    if IDX_TO_COMP[a] != env.compound:
                        valid_actions.append(a)
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_vals = [(a, Q[(state, a)]) for a in valid_actions]
                action = max(q_vals, key=lambda x: x[1])[0]
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Q-Learning update
            if not done:
                # Get max Q-value for next state
                next_valid = [0] if env.pitted else [0, 1, 2, 3]
                max_next_q = max([Q[(next_state, a)] for a in next_valid])
                Q[(state, action)] += alpha * (reward + gamma * max_next_q - Q[(state, action)])
            else:
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
            
            state = next_state
        
        if (ep + 1) % 1000 == 0:
            print(f"  Episode {ep + 1}/{episodes}: Total time = {-total_reward:.1f}s, ε = {epsilon:.3f}")
    
    return Q


def merge_q_tables(q_tables):
    """Merge multiple Q-tables into one comprehensive table"""
    merged = {}
    for Q in q_tables:
        merged.update(Q)
    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("F1 STRATEGY RL TRAINING - COMPREHENSIVE")
    print("=" * 60)
    
    # Train for each starting compound
    all_q_tables = []
    
    for start in COMPOUNDS:
        Q = train_qlearning(start, episodes=5000)
        all_q_tables.append(Q)
        print(f"✓ Completed training for {start} (Q-table entries: {len(Q)})")
    
    # Merge all Q-tables
    print("\nMerging Q-tables...")
    merged_q = merge_q_tables(all_q_tables)
    print(f"✓ Merged Q-table has {len(merged_q)} total entries")
    
    # Save the comprehensive Q-table
    os.makedirs("models", exist_ok=True)
    policy_data = {
        "q_table": merged_q,
        "bins": {"lap": LAP_BIN, "age": AGE_BIN}
    }
    
    with open("models/rl_policy.pkl", "wb") as f:
        pickle.dump(policy_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✓ Saved comprehensive RL policy to models/rl_policy.pkl")
    print(f"  Total Q-table entries: {len(merged_q)}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Restart your server to use the new RL policy.")
    print("=" * 60)
