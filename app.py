import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import streamlit as st

# Try gymnasium first, fall back to gym (API compatibility)
try:
    import gymnasium as gym
    from gymnasium import spaces
    USING_GYMNASIUM = True
except Exception:
    import gym
    from gym import spaces
    USING_GYMNASIUM = False

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(
    page_title="Deep Space Explorer - RL Simulation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #4da6ff; }
        .stProgress > div > div { background-color: #4da6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Environment --------------------
class DynamicSpaceExplorationEnv(gym.Env):
    """
    Custom 2D grid world with moving obstacles, gravity wells, fuel stations, and a goal.
    Observation (8,):
        [x_norm, y_norm, fuel_norm, gravity_effect, nearest_obstacle_dist_norm,
         nearest_fuel_dist_norm, goal_dx_norm, goal_dy_norm]
    Actions: Discrete(4) [left, right, up, down]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty: str = "medium"):
        super().__init__()

        # Actions: [left, right, up, down]
        self.action_space = spaces.Discrete(4)

        # Observation vector of 8 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.grid_size = 20
        self.state = np.zeros(8, dtype=np.float32)

        self.set_difficulty(difficulty)
        self.reset()

        self.total_rewards = []
        self.close_calls = 0
        self.refuels = 0

    def set_difficulty(self, difficulty: str):
        difficulty = difficulty.lower().strip()
        if difficulty == "easy":
            self.num_obstacles = 3
            self.num_gravity_wells = 1
            self.num_fuel_stations = 2
            self.obstacle_speed = 0.05
            self.max_steps = 300
            self.fuel_consumption_rate = 0.8
            self.initial_fuel = 100
        elif difficulty == "medium":
            self.num_obstacles = 5
            self.num_gravity_wells = 2
            self.num_fuel_stations = 2
            self.obstacle_speed = 0.1
            self.max_steps = 250
            self.fuel_consumption_rate = 1.0
            self.initial_fuel = 100
        else:  # hard
            self.num_obstacles = 7
            self.num_gravity_wells = 3
            self.num_fuel_stations = 1
            self.obstacle_speed = 0.15
            self.max_steps = 200
            self.fuel_consumption_rate = 1.2
            self.initial_fuel = 90

    # Gym/Gymnasium reset compatibility
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.trajectory = []
        self.reward_history = []
        self.actions_taken = []
        self.close_calls = 0
        self.refuels = 0

        # Start rocket bottom quarter
        self.rocket_pos = np.array(
            [
                np.random.randint(1, self.grid_size - 1),
                np.random.randint(1, max(2, self.grid_size // 4)),
            ],
            dtype=np.float32,
        )

        self.fuel = float(self.initial_fuel)

        # Goal top quarter
        self.goal = np.array(
            [
                np.random.randint(1, self.grid_size - 1),
                np.random.randint(max(1, (3 * self.grid_size) // 4), self.grid_size - 1),
            ],
            dtype=np.float32,
        )

        # Obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                obs_pos = np.array(
                    [
                        np.random.randint(1, self.grid_size - 1),
                        np.random.randint(
                            max(1, self.grid_size // 4), max(2, (3 * self.grid_size) // 4)
                        ),
                    ],
                    dtype=np.float32,
                )
                if (
                    np.linalg.norm(obs_pos - self.rocket_pos) > 3
                    and np.linalg.norm(obs_pos - self.goal) > 3
                ):
                    break

            angle = np.random.uniform(0, 2 * np.pi)
            vel = self.obstacle_speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            self.obstacles.append(
                {"position": obs_pos, "velocity": vel, "radius": float(np.random.uniform(0.5, 1.0))}
            )

        # Gravity wells
        self.gravity_wells = []
        for _ in range(self.num_gravity_wells):
            while True:
                grav_pos = np.array(
                    [
                        np.random.randint(1, self.grid_size - 1),
                        np.random.randint(
                            max(1, self.grid_size // 4), max(2, (3 * self.grid_size) // 4)
                        ),
                    ],
                    dtype=np.float32,
                )
                if (
                    np.linalg.norm(grav_pos - self.rocket_pos) > 4
                    and np.linalg.norm(grav_pos - self.goal) > 4
                ):
                    break

            strength = float(np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0))
            self.gravity_wells.append(
                {
                    "position": grav_pos,
                    "strength": strength,
                    "radius": float(np.random.uniform(1.5, 3.0)),
                }
            )

        # Fuel stations
        self.fuel_stations = []
        for _ in range(self.num_fuel_stations):
            while True:
                fuel_pos = np.array(
                    [
                        np.random.randint(1, self.grid_size - 1),
                        np.random.randint(
                            max(1, self.grid_size // 4), max(2, (3 * self.grid_size) // 4)
                        ),
                    ],
                    dtype=np.float32,
                )

                valid_pos = (
                    np.linalg.norm(fuel_pos - self.rocket_pos) > 3
                    and np.linalg.norm(fuel_pos - self.goal) > 3
                )
                if valid_pos:
                    for well in self.gravity_wells:
                        if np.linalg.norm(fuel_pos - well["position"]) < well["radius"]:
                            valid_pos = False
                            break
                if valid_pos:
                    break

            self.fuel_stations.append(
                {"position": fuel_pos, "refill_amount": int(np.random.randint(20, 50))}
            )

        self.trajectory.append(self.rocket_pos.copy())
        self._update_state()

        # Return shape depends on backend
        if USING_GYMNASIUM:
            return self.state, {}
        return self.state

    def _update_state(self):
        # nearest obstacle
        min_obstacle_dist = min(
            (np.linalg.norm(self.rocket_pos - o["position"]) for o in self.obstacles),
            default=float(self.grid_size),
        )
        # nearest fuel
        min_fuel_dist = min(
            (np.linalg.norm(self.rocket_pos - s["position"]) for s in self.fuel_stations),
            default=float(self.grid_size),
        )
        # gravity effect scalar [-1,1]
        gravity_effect = 0.0
        for w in self.gravity_wells:
            dist = np.linalg.norm(self.rocket_pos - w["position"])
            if dist < w["radius"]:
                gravity_effect += w["strength"] * (1 - dist / w["radius"])
        gravity_effect = float(np.clip(gravity_effect, -1, 1))

        goal_vec = self.goal - self.rocket_pos

        self.state[0] = self.rocket_pos[0] / self.grid_size
        self.state[1] = self.rocket_pos[1] / self.grid_size
        self.state[2] = np.clip(self.fuel / self.initial_fuel, 0, 1)
        self.state[3] = gravity_effect
        self.state[4] = min_obstacle_dist / self.grid_size
        self.state[5] = min_fuel_dist / self.grid_size
        self.state[6] = goal_vec[0] / self.grid_size
        self.state[7] = goal_vec[1] / self.grid_size

    def _move_obstacles(self):
        for obs in self.obstacles:
            obs["position"] += obs["velocity"]
            # Bounce
            for k in (0, 1):
                if obs["position"][k] <= 0 or obs["position"][k] >= self.grid_size - 1:
                    obs["velocity"][k] *= -1
                    obs["position"][k] = float(np.clip(obs["position"][k], 0, self.grid_size - 1))

    def _calculate_gravity_effect(self):
        force = np.zeros(2, dtype=np.float32)
        for w in self.gravity_wells:
            direction = w["position"] - self.rocket_pos
            dist = np.linalg.norm(direction)
            if dist > w["radius"]:
                continue
            if dist > 0:
                direction = direction / dist
            magnitude = w["strength"] * (1 - dist / w["radius"]) ** 2
            force += magnitude * direction
        return force

    def step(self, action: int):
        self.current_step += 1
        prev_pos = self.rocket_pos.copy()
        prev_dist_goal = float(np.linalg.norm(self.goal - prev_pos))

        self.actions_taken.append(int(action))
        # discrete move
        if action == 0:
            self.rocket_pos[0] -= 1
        elif action == 1:
            self.rocket_pos[0] += 1
        elif action == 2:
            self.rocket_pos[1] += 1
        elif action == 3:
            self.rocket_pos[1] -= 1

        # gravity nudge
        gravity_vec = self._calculate_gravity_effect()
        self.rocket_pos += gravity_vec * 0.5
        self.rocket_pos = np.clip(self.rocket_pos, 0, self.grid_size - 1)

        self._move_obstacles()
        self.trajectory.append(self.rocket_pos.copy())

        # fuel usage
        gravity_mag = float(np.linalg.norm(gravity_vec))
        self.fuel -= self.fuel_consumption_rate * (1 + 0.5 * gravity_mag)
        self.fuel = float(max(self.fuel, 0.0))

        # refuel
        refueled = False
        for station in list(self.fuel_stations):
            if np.linalg.norm(self.rocket_pos - station["position"]) < 1.0:
                old = self.fuel
                self.fuel = float(min(self.initial_fuel, self.fuel + station["refill_amount"]))
                if self.fuel > old:
                    self.refuels += 1
                    refueled = True
                self.fuel_stations.remove(station)
                break

        # reward shaping
        reward = -1.0
        # near-miss analytics
        for obs in self.obstacles:
            d = np.linalg.norm(self.rocket_pos - obs["position"])
            if 1.0 < d < 2.0:
                self.close_calls += 1
                break

        # collision
        collision = any(
            np.linalg.norm(self.rocket_pos - obs["position"]) < obs["radius"] for obs in self.obstacles
        )
        if collision:
            reward -= 50.0

        # progress reward
        dist_goal = float(np.linalg.norm(self.goal - self.rocket_pos))
        if dist_goal < prev_dist_goal:
            reward += 2.0 * (prev_dist_goal - dist_goal)

        if refueled:
            reward += 5.0

        # termination
        reached_goal = dist_goal < 1.0
        done = False
        if reached_goal:
            reward += 100.0 + 0.5 * self.fuel
            done = True
        elif collision:
            done = True
        elif self.fuel <= 0:
            reward -= 30.0
            done = True
        elif self.current_step >= self.max_steps:
            reward -= 20.0
            done = True

        self.reward_history.append(float(reward))
        self._update_state()

        info = {
            "fuel": float(self.fuel),
            "distance_to_goal": dist_goal,
            "gravity_effect": gravity_vec,
            "close_calls": int(self.close_calls),
            "refuels": int(self.refuels),
        }

        if USING_GYMNASIUM:
            # Gymnasium expects (obs, reward, terminated, truncated, info)
            terminated = done
            truncated = False
            return self.state, float(reward), terminated, truncated, info
        else:
            # Classic Gym: (obs, reward, done, info)
            return self.state, float(reward), done, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Rocket Position: {self.rocket_pos}, Fuel: {self.fuel:.1f}")
        return self.plot_state()

    def plot_state(self):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
        ax.set_facecolor("#040720")
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.grid(True, linestyle="-", alpha=0.15, color="gray")

        # stars
        num_stars = 200
        star_positions = np.random.rand(num_stars, 2) * self.grid_size
        star_sizes = np.random.exponential(0.7, num_stars)
        star_colors = np.random.choice(["white", "#FFFDD0", "#F8F7FF", "#CAE9FF"], num_stars)
        ax.scatter(star_positions[:, 0], star_positions[:, 1], color=star_colors, s=star_sizes, alpha=0.8, zorder=1)

        # nebulas
        for _ in range(3):
            nebula_pos = np.random.rand(2) * self.grid_size
            nebula_size = np.random.uniform(3, 5)
            nebula_color = np.random.choice(["purple", "blue", "pink"])
            nebula = plt.Circle(nebula_pos, nebula_size, color=nebula_color, alpha=0.05, zorder=1)
            ax.add_patch(nebula)

        # gravity wells
        for well in self.gravity_wells:
            cmap = plt.cm.Blues if well["strength"] > 0 else plt.cm.Reds
            num_circles = 12
            for i in range(num_circles):
                radius = well["radius"] * (i + 1) / num_circles
                color_val = cmap(0.7 - 0.5 * i / num_circles)
                circle = plt.Circle(well["position"], radius, color=color_val, alpha=0.4 * (1 - i / num_circles), zorder=2)
                ax.add_patch(circle)

            ax.scatter(well["position"][0], well["position"][1], color="white", s=50, zorder=3, edgecolor="black")
            sign = "+" if well["strength"] > 0 else "-"
            t = ax.text(well["position"][0], well["position"][1], sign, ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold", zorder=4)
            t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="black")])

        # fuel stations
        for station in self.fuel_stations:
            glow = plt.Circle(station["position"], 1.0, color="green", alpha=0.2, zorder=3)
            marker = plt.Circle(station["position"], 0.5, color="lime", alpha=0.8, zorder=4)
            ax.add_patch(glow)
            ax.add_patch(marker)
            t = ax.text(station["position"][0], station["position"][1], "F", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold", zorder=5)
            t.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])

        # obstacles
        for obs in self.obstacles:
            obstacle = plt.Circle(obs["position"], obs["radius"], color="red", alpha=0.7, zorder=5)
            ax.add_patch(obstacle)
            ring = plt.Circle(obs["position"], obs["radius"] + 0.3, color="red", alpha=0.2, fill=False, linestyle="--", zorder=5)
            ax.add_patch(ring)
            if np.linalg.norm(obs["velocity"]) > 0:
                ax.arrow(obs["position"][0], obs["position"][1],
                         obs["velocity"][0] * 3, obs["velocity"][1] * 3,
                         head_width=0.3, head_length=0.3, fc="red", ec="red", alpha=0.7, zorder=6)

        # goal
        goal_glow = plt.Circle(self.goal, 1.5, color="green", alpha=0.15, zorder=5)
        goal_outer = plt.Circle(self.goal, 1.0, color="green", alpha=0.4, zorder=6)
        goal_marker = plt.Circle(self.goal, 0.8, color="#00FF00", alpha=0.8, zorder=6)
        ax.add_patch(goal_glow)
        ax.add_patch(goal_outer)
        ax.add_patch(goal_marker)
        t = ax.text(self.goal[0], self.goal[1], "G", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold", zorder=7)
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="green")])

        # trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            colors = plt.cm.cool(np.linspace(0, 1, len(traj) - 1))
            for i in range(len(traj) - 1):
                seg = traj[i : i + 2]
                line = plt.Line2D(seg[:, 0], seg[:, 1], color=colors[i],
                                  linewidth=2, alpha=min(0.2 + i / (len(traj) - 1), 0.9), zorder=4)
                ax.add_line(line)

        # rocket
        engine_glow = plt.Circle(self.rocket_pos, 0.8, color="#00FFFF", alpha=0.2, zorder=7)
        rocket_marker = plt.Circle(self.rocket_pos, 0.6, color="cyan", alpha=1.0, zorder=8)
        ax.add_patch(engine_glow)
        ax.add_patch(rocket_marker)

        if len(self.trajectory) > 1:
            last_pos = self.trajectory[-2]
            movement = self.rocket_pos - last_pos
            n = np.linalg.norm(movement)
            if n > 0:
                movement = movement / n * 0.8
                ax.arrow(self.rocket_pos[0], self.rocket_pos[1],
                         movement[0], movement[1],
                         head_width=0.3, head_length=0.3, fc="white", ec="white", zorder=9)

        # HUD
        info_x, info_y = 0.02, 0.98
        line_h = 0.04
        fuel_ratio = self.fuel / self.initial_fuel if self.initial_fuel else 0.0
        fuel_color = "green" if fuel_ratio > 0.6 else "yellow" if fuel_ratio > 0.3 else "red"
        t = ax.text(info_x, info_y, f"FUEL: {self.fuel:.1f}", transform=ax.transAxes,
                    color=fuel_color, fontsize=12, va="top", fontweight="bold")
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="black")])

        t = ax.text(info_x, info_y - line_h, f"STEP: {self.current_step}/{self.max_steps}",
                    transform=ax.transAxes, color="white", fontsize=12, va="top")
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="black")])

        dist_to_goal = float(np.linalg.norm(self.goal - self.rocket_pos))
        t = ax.text(info_x, info_y - 2 * line_h, f"DISTANCE TO GOAL: {dist_to_goal:.1f}",
                    transform=ax.transAxes, color="white", fontsize=12, va="top")
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground="black")])

        title = ax.set_title("DEEP SPACE EXPLORER MISSION", color="#4da6ff", fontsize=18, fontweight="bold", pad=20)
        title.set_path_effects([path_effects.withStroke(linewidth=3, foreground="black")])

        ax.set_xlabel("X Position", color="#888888", fontsize=10)
        ax.set_ylabel("Y Position", color="#888888", fontsize=10)
        ax.tick_params(axis="x", colors="#888888")
        ax.tick_params(axis="y", colors="#888888")
        return fig

# -------------------- PPO-style A2C --------------------
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        x = self.shared(state)
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.99,
                 lr=0.001, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.training_log = []
        self.trajectory_buffer = []

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs, _ = self.model(state_t)
        dist = distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), probs.cpu().numpy()

    def evaluate(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs, value = self.model(state_t)
            action = torch.argmax(probs)
        return int(action.item()), float(value.item()), probs.cpu().numpy()

    def compute_returns(self, rewards, dones, values, next_value):
        returns, advantages = [], []
        advantage = 0.0
        ret = next_value
        for t in reversed(range(len(rewards))):
            non_term = 0.0 if dones[t] else 1.0
            ret = rewards[t] + self.gamma * non_term * ret
            returns.insert(0, ret)
            next_val = values[t + 1] if t + 1 < len(values) else next_value
            delta = rewards[t] + self.gamma * non_term * next_val - values[t]
            advantage = delta + self.gamma * 0.95 * non_term * advantage
            advantages.insert(0, advantage)
        return returns, advantages

    def train_minibatch(self, states, actions, old_log_probs, returns, advantages):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        probs, values = self.model(states)
        values = values.squeeze(-1)

        dist = distributions.Categorical(probs)
        curr_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (values - returns).pow(2).mean()
        entropy_loss = -self.entropy_coef * entropy

        loss = actor_loss + self.value_coef * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return float(loss.item())

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.trajectory_buffer.append(
            {"state": state, "action": action, "reward": float(reward),
             "next_state": next_state, "done": bool(done), "log_prob": float(log_prob)}
        )

    def train_on_buffer(self, next_value=0.0):
        states = [t["state"] for t in self.trajectory_buffer]
        actions = [t["action"] for t in self.trajectory_buffer]
        rewards = [t["reward"] for t in self.trajectory_buffer]
        dones = [t["done"] for t in self.trajectory_buffer]
        old_log_probs = [t["log_prob"] for t in self.trajectory_buffer]

        with torch.no_grad():
            values = []
            for s in states:
                _, v = self.model(torch.as_tensor(s, dtype=torch.float32, device=self.device))
                values.append(float(v.item()))

        returns, advantages = self.compute_returns(rewards, dones, values, next_value)
        # PPO-style epochs over same buffer
        for _ in range(4):
            _ = self.train_minibatch(states, actions, old_log_probs, returns, advantages)

        self.trajectory_buffer = []
        return

    def _step_env_compat(self, env, action):
        """Handle gym and gymnasium step tuple arity."""
        result = env.step(action)
        if USING_GYMNASIUM:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            return obs, float(reward), done, info
        else:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), info

    def train(self, env, episodes, update_freq=20, update_display=None):
        episode_rewards, episode_lengths = [], []
        for ep in range(episodes):
            reset_result = env.reset()
            state = reset_result[0] if USING_GYMNASIUM else reset_result
            ep_reward = 0.0
            self.trajectory_buffer = []

            done = False
            while not done:
                action, log_prob, _ = self.act(state)
                next_state, reward, done, _ = self._step_env_compat(env, action)
                ep_reward += reward
                self.store_transition(state, action, reward, next_state, done, log_prob)
                state = next_state

                if update_display and env.current_step % 5 == 0:
                    update_display(env, ep, ep_reward)

                if len(self.trajectory_buffer) >= update_freq or done:
                    next_value = 0.0
                    if not done:
                        _, next_value, _ = self.evaluate(next_state)
                    self.train_on_buffer(next_value)

            self.training_log.append({"episode": ep + 1, "reward": ep_reward, "steps": env.current_step})
            episode_rewards.append(ep_reward)
            episode_lengths.append(env.current_step)

            if update_display:
                update_display(env, ep, ep_reward, done=True)

        return episode_rewards, episode_lengths

    def save(self, path: str):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()},
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# -------------------- Streamlit App --------------------
def main():
    st.title("🚀 Dynamic Space Exploration with Reinforcement Learning")

    st.sidebar.header("Environment Settings")
    difficulty = st.sidebar.selectbox("Select Difficulty Level", ["easy", "medium", "hard"], index=1)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Training Controls")
        episodes = st.number_input("Number of Episodes", min_value=1, max_value=200, value=20)
        update_freq = st.number_input("Update Frequency", min_value=5, max_value=100, value=20)

        train_button = st.button("Start Training")
        test_button = st.button("Test Trained Agent")

        st.subheader("Agent Parameters")
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        gamma = st.slider("Discount Factor (Gamma)", min_value=0.80, max_value=0.999, value=0.99)

        st.subheader("Training Progress")
        episode_progress = st.empty()
        reward_text = st.empty()
        steps_text = st.empty()

        st.subheader("Training History")
        history_chart = st.empty()

    with col1:
        st.subheader("Environment Visualization")
        env_placeholder = st.empty()

    env = DynamicSpaceExplorationEnv(difficulty=difficulty)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size=state_size, action_size=action_size, hidden_size=128, gamma=gamma, lr=learning_rate)
    model_checkpoint = "trained_agent.pth"

    def update_display(env, episode, reward, done=False):
        fig = env.render(mode="human")
        env_placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)

        progress = (episode + 1) / max(1, episodes)
        episode_progress.progress(min(max(progress, 0.0), 1.0))
        reward_text.text(f"Current Episode Reward: {reward:.2f}")
        steps_text.text(f"Steps: {env.current_step}/{env.max_steps}")

        if len(agent.training_log) > 0:
            df = pd.DataFrame(agent.training_log).set_index("episode")
            history_chart.line_chart(df[["reward", "steps"]])

    if train_button:
        st.sidebar.info("Training in progress...")
        episode_rewards, episode_lengths = agent.train(
            env, int(episodes), update_freq=int(update_freq), update_display=update_display
        )
        agent.save(model_checkpoint)
        st.sidebar.success(f"Training completed! Model saved as {model_checkpoint}")

        st.subheader("Training Results")
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Average Reward", f"{np.mean(episode_rewards):.2f}")
            st.metric("Max Reward", f"{np.max(episode_rewards):.2f}")
        with c4:
            st.metric("Average Episode Length", f"{np.mean(episode_lengths):.1f}")
            success_rate = np.sum([r > 0 for r in episode_rewards]) / len(episode_rewards)
            st.metric("Success Rate", f"{success_rate:.1%}")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.plot(range(1, len(episode_rewards) + 1), episode_rewards)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Steps")
        ax2.plot(range(1, len(episode_lengths) + 1), episode_lengths, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if test_button:
        try:
            try:
                agent.load(model_checkpoint)
                st.sidebar.success("Loaded trained model successfully!")
            except Exception:
                st.sidebar.warning("No trained model found. Using untrained agent.")

            reset_result = env.reset()
            state = reset_result[0] if USING_GYMNASIUM else reset_result
            ep_reward = 0.0
            done = False

            test_progress = st.sidebar.progress(0.0)
            test_status = st.sidebar.empty()

            while not done:
                action, _, _ = agent.evaluate(state)
                next_state, reward, done, _ = agent._step_env_compat(env, action)
                ep_reward += reward

                if env.current_step % 3 == 0 or done:
                    update_display(env, 0, ep_reward)
                    test_progress.progress(min(env.current_step / env.max_steps, 1.0))
                    status = "Testing..."
                    if np.linalg.norm(env.rocket_pos - env.goal) < 1.0:
                        status = "Goal reached! 🎉"
                    elif env.fuel <= 0:
                        status = "Out of fuel! ⛽"
                    test_status.text(f"Status: {status} | Step: {env.current_step}")
                    time.sleep(0.05)

                state = next_state

            test_status.text(f"Test completed! Final reward: {ep_reward:.2f}")

        except Exception as e:
            st.error(f"Error during testing: {e}")

    with st.expander("About this application"):
        st.markdown(
            """
            **Dynamic Space Exploration with Reinforcement Learning**

            The agent (rocket) must reach the goal while avoiding moving obstacles,
            managing fuel, and dealing with gravitational wells.

            - Obstacles (red) move and can collide with you.
            - Gravity wells (blue = attractive, red = repulsive).
            - Fuel stations (green 'F') can refill fuel.
            - Reward shaping encourages progress toward the goal.

            Controls:
            1. Pick a difficulty.
            2. Adjust agent parameters if desired.
            3. Train, then Test.
            """
        )

if __name__ == "__main__":
    main()
