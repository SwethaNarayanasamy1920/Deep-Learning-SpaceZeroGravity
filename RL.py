import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import pandas as pd
import time
import math
from matplotlib.patches import Rectangle, Circle, Wedge
from gym import spaces
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Deep Space Explorer - RL Simulation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    /* General body styling for a space theme */
    body {
        background-color: #0d1117; /* Dark GitHub-like background */
        color: #e6edf3; /* Light text color */
        font-family: 'Segoe UI', sans-serif;
    }
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff; /* A vibrant blue for headers */
        font-family: 'Orbitron', sans-serif; /* A sci-fi font if available, or a good sans-serif fallback */
        text-shadow: 1px 1px 5px rgba(88, 166, 255, 0.4);
    }
    .stButton>button {
        background-color: #2188ff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        box-shadow: 0 0 15px rgba(33, 136, 255, 0.6);
    }
    .stProgress > div > div > div > div {
        background-color: #2188ff;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
    }
    /* Card-like containers for sections - adjust selector if needed after Streamlit updates */
    .st-emotion-cache-nahz7x { /* This class might change, inspect with dev tools if issues */
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border: 1px solid #30363d;
    }
    /* Specific styling for the plot area background to match */
    .stPlotlyChart {
        background-color: #161b22; /* Matches card background */
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Define Custom Dynamic Space Exploration Environment
class DynamicSpaceExplorationEnv(gym.Env):
    def __init__(self, difficulty='medium'):
        super(DynamicSpaceExplorationEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: [left, right, up, down]
        
        # State features: [x, y, fuel, gravity_effect, nearest_obstacle_dist, fuel_station_dist, goal_x_dist, goal_y_dist]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Grid parameters
        self.grid_size = 20
        self.state = np.zeros(8, dtype=np.float32)
        
        # Set difficulty parameters
        self.set_difficulty(difficulty)
        
        # Initialize components
        self.reset()
        
        # Performance metrics
        self.total_rewards = []
        self.close_calls = 0
        self.refuels = 0
        
    def set_difficulty(self, difficulty):
        if difficulty == 'easy':
            self.num_obstacles = 3
            self.num_gravity_wells = 1
            self.num_fuel_stations = 2
            self.obstacle_speed = 0.05
            self.max_steps = 300
            self.fuel_consumption_rate = 0.8
            self.initial_fuel = 100
        elif difficulty == 'medium':
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
    
    def reset(self):
        # Reset step counter and trajectory
        self.current_step = 0
        self.trajectory = []
        self.reward_history = []
        self.actions_taken = []
        self.close_calls = 0
        self.refuels = 0
        
        # Place rocket at a random starting position in the bottom quarter of the grid
        self.rocket_pos = np.array([
            np.random.randint(1, self.grid_size-1),
            np.random.randint(1, int(self.grid_size/4))
        ], dtype=np.float32)
        
        # Set initial fuel
        self.fuel = self.initial_fuel
        
        # Place goal at a random position in the top quarter of the grid
        self.goal = np.array([
            np.random.randint(1, self.grid_size-1),
            np.random.randint(int(3*self.grid_size/4), self.grid_size-1)
        ], dtype=np.float32)
        
        # Initialize obstacles with random positions and velocities
        self.obstacles = []
        for _ in range(self.num_obstacles):
            # Ensure obstacles don't start at rocket or goal positions
            while True:
                obs_pos = np.array([
                    np.random.randint(1, self.grid_size-1),
                    np.random.randint(int(self.grid_size/4), int(3*self.grid_size/4))
                ], dtype=np.float32)
                
                if (np.linalg.norm(obs_pos - self.rocket_pos) > 3 and 
                    np.linalg.norm(obs_pos - self.goal) > 3):
                    break
            
            # Random velocity - direction and magnitude
            angle = np.random.uniform(0, 2*np.pi)
            vel_x = self.obstacle_speed * np.cos(angle)
            vel_y = self.obstacle_speed * np.sin(angle)
            
            # Add obstacle with position and velocity
            self.obstacles.append({
                'position': obs_pos,
                'velocity': np.array([vel_x, vel_y], dtype=np.float32),
                'radius': np.random.uniform(0.5, 1.0)
            })
        
        # Initialize gravity wells
        self.gravity_wells = []
        for _ in range(self.num_gravity_wells):
            # Ensure gravity wells don't start at rocket or goal positions
            while True:
                grav_pos = np.array([
                    np.random.randint(1, self.grid_size-1),
                    np.random.randint(int(self.grid_size/4), int(3*self.grid_size/4))
                ], dtype=np.float32)
                
                if (np.linalg.norm(grav_pos - self.rocket_pos) > 4 and 
                    np.linalg.norm(grav_pos - self.goal) > 4):
                    break
            
            # Random strength (positive: attractive, negative: repulsive)
            strength = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
            
            # Add gravity well with position and strength
            self.gravity_wells.append({
                'position': grav_pos,
                'strength': strength,
                'radius': np.random.uniform(1.5, 3.0)
            })
        
        # Initialize fuel stations
        self.fuel_stations = []
        for _ in range(self.num_fuel_stations):
            # Ensure fuel stations don't start at rocket, goal, or gravity well positions
            while True:
                fuel_pos = np.array([
                    np.random.randint(1, self.grid_size-1),
                    np.random.randint(int(self.grid_size/4), int(3*self.grid_size/4))
                ], dtype=np.float32)
                
                valid_pos = (np.linalg.norm(fuel_pos - self.rocket_pos) > 3 and 
                             np.linalg.norm(fuel_pos - self.goal) > 3)
                
                for well in self.gravity_wells:
                    if np.linalg.norm(fuel_pos - well['position']) < well['radius']:
                        valid_pos = False
                        break
                
                if valid_pos:
                    break
            
            # Add fuel station with position and refill amount
            self.fuel_stations.append({
                'position': fuel_pos,
                'refill_amount': np.random.randint(20, 50)
            })
        
        # Store initial state
        self.trajectory.append(self.rocket_pos.copy())
        
        # Update state vector
        self._update_state()
        
        return self.state
    
    def _update_state(self):
        """Update the state vector with current environment state"""
        # Calculate distance to nearest obstacle
        min_obstacle_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(self.rocket_pos - obs['position'])
            min_obstacle_dist = min(min_obstacle_dist, dist)
        
        if min_obstacle_dist == float('inf'):
            min_obstacle_dist = self.grid_size  # No obstacles
            
        # Calculate distance to nearest fuel station
        min_fuel_dist = float('inf')
        for station in self.fuel_stations:
            dist = np.linalg.norm(self.rocket_pos - station['position'])
            min_fuel_dist = min(min_fuel_dist, dist)
            
        if min_fuel_dist == float('inf'):
            min_fuel_dist = self.grid_size  # No fuel stations
        
        # Calculate current gravity effect
        gravity_effect = 0
        for well in self.gravity_wells:
            dist = np.linalg.norm(self.rocket_pos - well['position'])
            if dist < well['radius']:
                # Gravity effect increases as distance decreases
                gravity_effect += well['strength'] * (1 - dist/well['radius'])
        
        # Normalize gravity effect to range [-1, 1]
        gravity_effect = np.clip(gravity_effect, -1, 1)
        
        # Calculate vector to goal
        goal_vector = self.goal - self.rocket_pos
        
        # Update state
        self.state[0] = self.rocket_pos[0] / self.grid_size  # Normalized x position
        self.state[1] = self.rocket_pos[1] / self.grid_size  # Normalized y position
        self.state[2] = self.fuel / self.initial_fuel  # Normalized fuel
        self.state[3] = gravity_effect  # Current gravity effect
        self.state[4] = min_obstacle_dist / self.grid_size  # Normalized distance to nearest obstacle
        self.state[5] = min_fuel_dist / self.grid_size  # Normalized distance to nearest fuel station
        self.state[6] = goal_vector[0] / self.grid_size  # Normalized x distance to goal
        self.state[7] = goal_vector[1] / self.grid_size  # Normalized y distance to goal
    
    def _move_obstacles(self):
        """Update positions of all dynamic obstacles"""
        for obs in self.obstacles:
            # Update position based on velocity
            obs['position'] += obs['velocity']
            
            # Bounce off boundaries
            if obs['position'][0] <= 0 or obs['position'][0] >= self.grid_size - 1:
                obs['velocity'][0] *= -1
                # Ensure obstacle stays in bounds
                obs['position'][0] = np.clip(obs['position'][0], 0, self.grid_size - 1)
                
            if obs['position'][1] <= 0 or obs['position'][1] >= self.grid_size - 1:
                obs['velocity'][1] *= -1
                # Ensure obstacle stays in bounds
                obs['position'][1] = np.clip(obs['position'][1], 0, self.grid_size - 1)
    
    def _calculate_gravity_effect(self):
        """Calculate gravitational force and direction from all gravity wells"""
        force_vector = np.zeros(2, dtype=np.float32)
        
        for well in self.gravity_wells:
            # Vector from rocket to gravity well
            direction = well['position'] - self.rocket_pos
            distance = np.linalg.norm(direction)
            
            # Skip if too far
            if distance > well['radius']:
                continue
                
            # Normalize direction
            if distance > 0:
                direction = direction / distance
                
            # Calculate force magnitude (inverse square law)
            magnitude = well['strength'] * (1 - distance/well['radius'])**2
            
            # Add to force vector
            force_vector += magnitude * direction
            
        return force_vector
    
    def step(self, action):
        self.current_step += 1
        prev_position = self.rocket_pos.copy()
        prev_distance_to_goal = np.linalg.norm(self.goal - prev_position)
        
        # Track action for analytics
        self.actions_taken.append(action)
        
        # Apply action
        if action == 0:  # Move left
            self.rocket_pos[0] -= 1
        elif action == 1:  # Move right
            self.rocket_pos[0] += 1
        elif action == 2:  # Move up
            self.rocket_pos[1] += 1
        elif action == 3:  # Move down
            self.rocket_pos[1] -= 1
            
        # Apply gravitational effects (subtle nudge)
        gravity_vector = self._calculate_gravity_effect()
        self.rocket_pos += gravity_vector * 0.5
        
        # Ensure rocket stays within grid boundaries
        self.rocket_pos = np.clip(self.rocket_pos, 0, self.grid_size - 1)
        
        # Move obstacles
        self._move_obstacles()
        
        # Update trajectory
        self.trajectory.append(self.rocket_pos.copy())
        
        # Calculate fuel consumption based on gravity and action
        gravity_magnitude = np.linalg.norm(gravity_vector)
        # More fuel needed to counteract stronger gravity
        self.fuel -= self.fuel_consumption_rate * (1 + 0.5 * gravity_magnitude)
        
        # Check for fuel station refill
        refueled = False
        for station in list(self.fuel_stations):  # Create a copy for safe removal
            if np.linalg.norm(self.rocket_pos - station['position']) < 1.0:
                old_fuel = self.fuel
                self.fuel = min(self.initial_fuel, self.fuel + station['refill_amount'])
                if self.fuel > old_fuel:  # Only count if fuel was actually added
                    self.refuels += 1
                    refueled = True
                # Remove the fuel station after use
                self.fuel_stations.remove(station)
                break
        
        # Calculate reward
        reward = -1  # Base step penalty
        
        # Check for close calls with obstacles (for analytics)
        for obs in self.obstacles:
            dist = np.linalg.norm(self.rocket_pos - obs['position'])
            if 1.0 < dist < 2.0:  # Close call but not collision
                self.close_calls += 1
                break
        
        # Check for collisions with obstacles
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(self.rocket_pos - obs['position']) < obs['radius']:
                reward -= 50  # Penalty for collision
                collision = True
                break
        
        # Check if reached goal
        reached_goal = np.linalg.norm(self.rocket_pos - self.goal) < 1.0
        
        # Reward for moving closer to goal
        current_distance_to_goal = np.linalg.norm(self.goal - self.rocket_pos)
        if current_distance_to_goal < prev_distance_to_goal:
            # More reward for bigger improvements
            reward += 2 * (prev_distance_to_goal - current_distance_to_goal)
        
        # Small bonus for successful refueling
        if refueled:
            reward += 5
        
        # Check termination conditions
        done = False
        
        if reached_goal:
            reward += 100 + self.fuel * 0.5  # Big reward for reaching goal + bonus for remaining fuel
            done = True
        elif collision:
            done = True
        elif self.fuel <= 0:
            reward -= 30  # Penalty for running out of fuel
            done = True
        elif self.current_step >= self.max_steps:
            reward -= 20  # Penalty for timeout
            done = True
        
        # Store reward for history
        self.reward_history.append(reward)
        
        # Update state
        self._update_state()
        
        # Extra info
        info = {
            'fuel': self.fuel,
            'distance_to_goal': current_distance_to_goal,
            'gravity_effect': gravity_vector,
            'close_calls': self.close_calls,
            'refuels': self.refuels
        }
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Rocket Position: {self.rocket_pos}, Fuel: {self.fuel:.1f}")
        return self.plot_state()
    
    def plot_state(self):
        """Generate a visual representation of the current state"""
        # Create figure with dark space background
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('#040720')  # Dark blue space color
        
        # Create grid
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        
        # Draw subtle grid lines
        ax.grid(True, linestyle='-', alpha=0.15, color='gray')
        
        # Create space background
        # Add stars in the background
        num_stars = 200
        star_positions = np.random.rand(num_stars, 2) * self.grid_size
        star_sizes = np.random.exponential(0.7, num_stars)  # More realistic star size distribution
        star_colors = np.random.choice(['white', '#FFFDD0', '#F8F7FF', '#CAE9FF'], num_stars)  # Different star colors
        
        ax.scatter(star_positions[:, 0], star_positions[:, 1], 
                  color=star_colors, s=star_sizes, alpha=0.8, zorder=1)
        
        # Add a few nebulas in the background for visual interest
        for _ in range(3):
            nebula_pos = np.random.rand(2) * self.grid_size
            nebula_size = np.random.uniform(3, 5)
            nebula_color = np.random.choice(['purple', 'blue', 'pink'])
            nebula = plt.Circle(nebula_pos, nebula_size, color=nebula_color, alpha=0.05, zorder=1)
            ax.add_patch(nebula)
        
        # Draw gravity wells
        for well in self.gravity_wells:
            # Use different colors for attractive vs repulsive
            if well['strength'] > 0:  # Attractive (blue)
                color = 'blue'
                cmap = plt.cm.Blues
            else:  # Repulsive (red)
                color = 'red'
                cmap = plt.cm.Reds
            
            # Draw concentric circles with fading alpha to show field strength
            num_circles = 12
            for i in range(num_circles):
                radius = well['radius'] * (i+1)/num_circles
                alpha = 0.4 * (1 - i/num_circles)  # Fade out with distance
                color_val = cmap(0.7 - 0.5 * i/num_circles)
                circle = plt.Circle(well['position'], radius, color=color_val, 
                                   alpha=alpha, fill=True, zorder=2)
                ax.add_patch(circle)
            
            # Add center point and label
            ax.scatter(well['position'][0], well['position'][1], 
                      color='white', s=50, zorder=3, edgecolor=color)
            
            # Add "+" or "-" label depending on gravity type
            sign = "+" if well['strength'] > 0 else "−"
            text = ax.text(well['position'][0], well['position'][1], sign, 
                   ha='center', va='center', color='white', fontsize=12, 
                   fontweight='bold', zorder=4)
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Draw fuel stations
        for station in self.fuel_stations:
            # Draw fuel station with glowing effect
            glow = plt.Circle(station['position'], 1.0, color='green', alpha=0.2, zorder=3)
            station_marker = plt.Circle(station['position'], 0.5, color='lime', alpha=0.8, zorder=4)
            ax.add_patch(glow)
            ax.add_patch(station_marker)
            
            # Add "F" label
            text = ax.text(station['position'][0], station['position'][1], 'F', 
                   ha='center', va='center', color='white', fontsize=10, 
                   fontweight='bold', zorder=5)
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
        
        # Draw obstacles
        for obs in self.obstacles:
            # Main obstacle
            obstacle = plt.Circle(obs['position'], obs['radius'], color='red', alpha=0.7, zorder=5)
            ax.add_patch(obstacle)
            
            # Add "danger" ring
            danger_ring = plt.Circle(obs['position'], obs['radius'] + 0.3, 
                                    color='red', alpha=0.2, fill=False, 
                                    linestyle='--', zorder=5)
            ax.add_patch(danger_ring)
            
            # Show velocity vector
            if np.linalg.norm(obs['velocity']) > 0:
                ax.arrow(obs['position'][0], obs['position'][1], 
                         obs['velocity'][0]*3, obs['velocity'][1]*3, 
                         head_width=0.3, head_length=0.3, fc='red', ec='red', 
                         alpha=0.7, zorder=6)
        
        # Draw goal with glowing effect
        goal_glow = plt.Circle(self.goal, 1.5, color='green', alpha=0.15, zorder=5)
        goal_outer = plt.Circle(self.goal, 1.0, color='green', alpha=0.4, zorder=6)
        goal_marker = plt.Circle(self.goal, 0.8, color='#00FF00', alpha=0.8, zorder=6)
        ax.add_patch(goal_glow)
        ax.add_patch(goal_outer)
        ax.add_patch(goal_marker)
        
        # Add "G" label
        text = ax.text(self.goal[0], self.goal[1], 'G', 
               ha='center', va='center', color='white', fontsize=12, 
               fontweight='bold', zorder=7)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='green')])
        
        # Draw trajectory with gradient color based on time
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            # Create color gradient for trajectory
            points = np.array([traj[:-1], traj[1:]]).transpose(1, 0, 2)
            segments = np.concatenate([points[i] for i in range(len(points))]).reshape(-1, 2, 2)
            
            # Color gradient from blue to cyan
            colors = plt.cm.cool(np.linspace(0, 1, len(segments)))
            
            for i, (segment, color) in enumerate(zip(segments, colors)):
                line = plt.Line2D(segment[:, 0], segment[:, 1], color=color, 
                                 linewidth=2, alpha=min(0.2 + i/len(segments), 0.9), zorder=4)
                ax.add_line(line)
        
        # Draw rocket with glowing effect
        engine_glow = plt.Circle(self.rocket_pos, 0.8, color='#00FFFF', alpha=0.2, zorder=7)
        rocket_marker = plt.Circle(self.rocket_pos, 0.6, color='cyan', alpha=1.0, zorder=8)
        ax.add_patch(engine_glow)
        ax.add_patch(rocket_marker)
        
        # Add rocket direction indicator
        if len(self.trajectory) > 1:
            last_pos = self.trajectory[-2]
            movement = self.rocket_pos - last_pos
            if np.linalg.norm(movement) > 0:
                movement = movement / np.linalg.norm(movement) * 0.8
                ax.arrow(self.rocket_pos[0], self.rocket_pos[1], 
                         movement[0], movement[1], 
                         head_width=0.3, head_length=0.3, fc='white', ec='white', zorder=9)
        
        # Add information panel in the corner
        info_x, info_y = 0.02, 0.98
        line_height = 0.04
        
        # Add fuel gauge with color based on fuel level
        fuel_ratio = self.fuel / self.initial_fuel
        fuel_color = 'green' if fuel_ratio > 0.6 else 'yellow' if fuel_ratio > 0.3 else 'red'
        fuel_text = f"FUEL: {self.fuel:.1f}"
        fuel_text_obj = ax.text(info_x, info_y, fuel_text, transform=ax.transAxes, 
                color=fuel_color, fontsize=12, verticalalignment='top', fontweight='bold')
        fuel_text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Step counter
        step_text = f"STEP: {self.current_step}/{self.max_steps}"
        step_text_obj = ax.text(info_x, info_y - line_height, step_text, transform=ax.transAxes, 
                color='white', fontsize=12, verticalalignment='top')
        step_text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Distance to goal
        dist_to_goal = np.linalg.norm(self.goal - self.rocket_pos)
        dist_text = f"DISTANCE TO GOAL: {dist_to_goal:.1f}"
        dist_text_obj = ax.text(info_x, info_y - 2*line_height, dist_text, transform=ax.transAxes, 
                color='white', fontsize=12, verticalalignment='top')
        dist_text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Title with space theme
        title = ax.set_title("DEEP SPACE EXPLORER MISSION", color='#4da6ff', fontsize=18, fontweight='bold', pad=20)
        title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
        
        # Subtle axis labels
        x_label = ax.set_xlabel("X Position", color='#888888', fontsize=10)
        y_label = ax.set_ylabel("Y Position", color='#888888', fontsize=10)
        
        # Make tick labels light gray for better visibility
        ax.tick_params(axis='x', colors='#888888')
        ax.tick_params(axis='y', colors='#888888')
        
        return fig


# Advantage Actor-Critic (A2C) Implementation with PPO improvements
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor with deeper network
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_probs, value


# PPO-enhanced A2C Agent
class A2CAgent:
    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.99, 
                 lr=0.001, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.clip_ratio = clip_ratio  # PPO clip ratio
        self.value_coef = value_coef  # Value loss coefficient
        self.entropy_coef = entropy_coef  # Entropy bonus coefficient
        
        # Initialize neural network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training logs
        self.training_log = []
        self.trajectory_buffer = []
        self.episode_metrics = {'successes': 0, 'collisions': 0, 'timeouts': 0, 'fuel_outs': 0}
        self.training_iterations = 0
    
    def act(self, state):
        """Select action based on current policy"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        
        # Sample action from probability distribution
        dist = distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), action_probs.cpu().numpy()
    
    def evaluate(self, state):
        """Evaluate state without sampling (for testing)"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, value = self.model(state)
            action = torch.argmax(action_probs)
        
        return action.item(), value.item(), action_probs.cpu().numpy()
    
    def compute_returns(self, rewards, dones, values, next_value):
        """Compute returns and advantages using GAE"""
        returns = []
        advantages = []
        advantage = 0
        next_return = next_value
        
        for t in reversed(range(len(rewards))):
            # Calculate return (discounted sum of future rewards)
            next_non_terminal = 1.0 - dones[t]
            next_return = rewards[t] + self.gamma * next_non_terminal * next_return
            returns.insert(0, next_return)
            
            # Calculate advantage
            next_value = values[t] if t < len(values) - 1 else next_value
            delta = rewards[t] + self.gamma * next_non_terminal * next_value - values[t]
            advantage = delta + self.gamma * 0.95 * next_non_terminal * advantage
            advantages.insert(0, advantage)
            
        return returns, advantages
    
    def train_minibatch(self, states, actions, old_log_probs, returns, advantages):
        """Train on a single minibatch of experience"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current action probabilities and state values
        action_probs, values = self.model(states)
        values = values.squeeze(-1)
        
        # Calculate log probabilities and entropy
        dist = distributions.Categorical(action_probs)
        curr_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Compute probability ratio and clipped ratio
        ratio = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # Calculate losses
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Combined loss
        loss = actor_loss + self.value_coef * critic_loss + entropy_loss
        
        # Perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Clip gradients
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition in buffer"""
        self.trajectory_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def train_on_buffer(self, next_value=0):
        """Train on collected buffer data"""
        # Extract data from buffer
        states = [t['state'] for t in self.trajectory_buffer]
        actions = [t['action'] for t in self.trajectory_buffer]
        rewards = [t['reward'] for t in self.trajectory_buffer]
        next_states = [t['next_state'] for t in self.trajectory_buffer]
        dones = [t['done'] for t in self.trajectory_buffer]
        old_log_probs = [t['log_prob'] for t in self.trajectory_buffer]
        
        # Compute value estimates for all states
        with torch.no_grad():
            values = []
            for state in states:
                _, value = self.model(torch.FloatTensor(state).to(self.device))
                values.append(value.item())
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns(rewards, dones, values, next_value)
        
        # Train for multiple epochs (PPO style)
        n_epochs = 4
        batch_size = len(self.trajectory_buffer)
        
        loss_info = []
        for _ in range(n_epochs):
            # Train on entire buffer
            loss, actor_loss, critic_loss, entropy = self.train_minibatch(
                states, actions, old_log_probs, returns, advantages
            )
            loss_info.append((loss, actor_loss, critic_loss, entropy))
        
        # Clear buffer after training
        avg_loss = np.mean([l[0] for l in loss_info])
        self.trajectory_buffer = []
        
        return avg_loss
    
    def train(self, env, episodes, update_freq=20, update_display=None):
        """Train the agent for a given number of episodes"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Reset buffer for PPO at the start of episode
            self.trajectory_buffer = []
            
            while not done:
                # Select action
                action, log_prob, _ = self.act(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done, log_prob)
                
                # Update state
                state = next_state
                
                # Update display if provided
                if update_display and env.current_step % 5 == 0:
                    update_display(env, episode, episode_reward)
                
                # Train if buffer filled or episode ended
                if len(self.trajectory_buffer) >= update_freq or done:
                    # If episode continues, estimate next state value
                    next_value = 0
                    if not done:
                        _, next_value, _ = self.evaluate(next_state)
                    
                    # Train on collected buffer
                    avg_loss = self.train_on_buffer(next_value)
            
            # Log episode stats
            self.training_log.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': env.current_step,
            })
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(env.current_step)

            # Update final display if provided
            if update_display:
                update_display(env, episode, episode_reward, done=True)
        
        return episode_rewards, episode_lengths
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_load_state_dict(checkpoint['optimizer_state_dict'])


# Streamlit Application
def main():
    st.title("🚀 Dynamic Space Exploration with Reinforcement Learning")
    
    # Create sidebar for configuration
    st.sidebar.header("Environment Settings")
    difficulty = st.sidebar.selectbox(
        "Select Difficulty Level",
        ["easy", "medium", "hard"],
        index=1
    )
    
    # Create columns for environment visualization and training metrics
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Training Controls")
        episodes = st.number_input("Number of Episodes", min_value=1, max_value=200, value=50)
        update_freq = st.number_input("Update Frequency", min_value=5, max_value=50, value=20)
        
        train_button = st.button("Start Training")
        test_button = st.button("Test Trained Agent")
        
        st.subheader("Agent Parameters")
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        gamma = st.slider("Discount Factor (Gamma)", min_value=0.8, max_value=0.999, value=0.99)
        
        st.subheader("Advanced Agent Hyperparameters")
        hidden_size = st.slider("Network Hidden Size", min_value=32, max_value=256, value=128, step=32)
        clip_ratio = st.slider("PPO Clip Ratio", min_value=0.05, max_value=0.5, value=0.2, format="%.2f")
        value_coef = st.slider("Value Loss Coefficient", min_value=0.1, max_value=1.0, value=0.5, format="%.1f")
        entropy_coef = st.slider("Entropy Coefficient", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")

        st.subheader("Training Progress")
        episode_progress = st.empty()
        reward_text = st.empty()
        steps_text = st.empty()
        
    with col1:
        st.subheader("Environment Visualization")
        env_placeholder = st.empty()
        st.subheader("Training History")
        history_chart_rewards = st.empty()
        history_chart_steps = st.empty()


    # Initialize environment and agent
    env = DynamicSpaceExplorationEnv(difficulty=difficulty)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = A2CAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        gamma=gamma,
        lr=learning_rate,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef
    )
    
    # Initialize model checkpoint
    model_checkpoint = "trained_agent.pth"
    
    # Function to update training display
    def update_display(env_instance, episode, reward, done=False):
        fig = env_instance.render(mode='human')
        env_placeholder.pyplot(fig)
        plt.close(fig)
        
        episode_progress.progress(min((episode + 1) / episodes, 1.0))
        reward_text.text(f"Current Episode Reward: {reward:.2f}")
        steps_text.text(f"Steps: {env_instance.current_step}/{env_instance.max_steps}")
        
        # Update training history charts
        if len(agent.training_log) > 0:
            df = pd.DataFrame(agent.training_log)
            
            fig_rewards = px.line(df, x='episode', y='reward', 
                                  title='Episode Rewards Over Time', 
                                  labels={'reward': 'Total Reward', 'episode': 'Episode'},
                                  color_discrete_sequence=["#58a6ff"])
            fig_rewards.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3",
                xaxis_title="Episode", yaxis_title="Total Reward",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            history_chart_rewards.plotly_chart(fig_rewards, use_container_width=True)

            fig_steps = px.line(df, x='episode', y='steps', 
                                title='Episode Lengths Over Time', 
                                labels={'steps': 'Steps Taken', 'episode': 'Episode'},
                                color_discrete_sequence=["#2ea043"])
            fig_steps.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3",
                xaxis_title="Episode", yaxis_title="Steps Taken",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            history_chart_steps.plotly_chart(fig_steps, use_container_width=True)


    # Training loop
    if train_button:
        st.sidebar.text("Training in progress...")
        
        # Train the agent
        with st.spinner("Training agent... This may take a while."):
            episode_rewards, episode_lengths = agent.train(
                env, episodes, update_freq=update_freq, update_display=update_display
            )
        
        # Save the trained model
        agent.save(model_checkpoint)
        
        st.sidebar.success(f"Training completed! Model saved as {model_checkpoint}")
        
        # Display final training results
        st.markdown("---") # Separator
        st.subheader("Mission Performance Summary")
        
        # Calculate success rate and other metrics based on reward thresholds
        # A simple heuristic for success: positive reward and goal reached (implied by high reward)
        successful_episodes = [r for r in episode_rewards if r > 50] 
        success_rate = len(successful_episodes) / episodes if episodes > 0 else 0

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Avg. Reward (Last 5 Episodes)", f"{np.mean(episode_rewards[-5:]):.2f}" if len(episode_rewards) >=5 else f"{np.mean(episode_rewards):.2f}")
        with col_kpi2:
            st.metric("Overall Success Rate", f"{success_rate:.1%}")
        with col_kpi3:
            # Note: total refuels and close calls are from the LAST episode of training.
            # To get aggregate, you'd need to modify agent.train to sum these across episodes.
            st.metric("Total Refuels (Last Ep)", f"{env.refuels}") 
        with col_kpi4:
            st.metric("Total Close Calls (Last Ep)", f"{env.close_calls}") 
        
    
    # Test trained agent
    if test_button:
        try:
            # Try to load trained model
            try:
                agent.load(model_checkpoint)
                st.sidebar.success("Loaded trained model successfully!")
            except Exception as e:
                st.sidebar.warning(f"No trained model found or error loading: {e}. Using untrained agent.")
            
            # Reset environment for testing
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Progress bar for testing
            test_progress = st.sidebar.progress(0)
            test_status = st.sidebar.empty()
            
            with st.spinner("Testing agent..."):
                # Run test episode
                while not done:
                    # Select best action (deterministic policy)
                    action, _, _ = agent.evaluate(state)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # Update display
                    if env.current_step % 3 == 0 or done:
                        # Pass current env instance to update_display
                        update_display(env, 0, episode_reward) 
                        
                        # Update test progress
                        test_progress.progress(min(env.current_step / env.max_steps, 1.0))
                        status = "Goal reached! 🎉" if np.linalg.norm(env.rocket_pos - env.goal) < 1.0 else "Testing..."
                        if env.fuel <= 0:
                            status = "Out of fuel! ⛽"
                        elif env.current_step >= env.max_steps:
                            status = "Timeout! ⏳"
                        test_status.markdown(f"Status: **{status}** | Step: **{env.current_step}**")
                        
                        # Add small delay for better visualization
                        time.sleep(0.1)
                    
                    # Update state
                    state = next_state
                
            # Final test results
            test_status.markdown(f"Test completed! Final reward: **{episode_reward:.2f}**")
            
        except Exception as e:
            st.error(f"Error during testing: {e}")
    
    # Add description and instructions with tabs
    with st.expander("About Deep Space Explorer", expanded=True):
        st.markdown("""
        ## Dynamic Space Exploration with Reinforcement Learning 🌌
        """)
        tab1, tab2, tab3 = st.tabs(["Overview", "Environment Details", "RL Agent"])

        with tab1:
            st.markdown("""
            This interactive application showcases a **Deep Reinforcement Learning (DRL)** agent navigating a challenging, dynamic space environment. The primary goal of the agent (rocket) is to reach a designated target while overcoming various perils and managing critical resources like fuel.
            
            **Key Challenge**: The agent must learn optimal strategies to balance reaching the goal, avoiding collisions, efficient fuel management, and leveraging or mitigating gravitational forces.
            
            **Your Mission**: Observe the agent learn and adapt, or test a pre-trained policy to see its capabilities!
            """)

        with tab2:
            st.markdown("""
            ### The Dynamic Space Environment:
            The environment is a grid-based simulation with several dynamic elements:
            * 🚀 **Rocket (Cyan)**: The agent's controlled entity. Its position, fuel, and proximity to environmental features define its state.
            * 🎯 **Goal (Green 'G')**: The target destination for the rocket. Reaching it yields significant rewards.
            * 🔴 **Moving Obstacles**: Red circular hazards that move randomly. Collision results in a large penalty and episode termination.
            * 🕳️ **Gravity Wells (Blue/Red Gradients)**:
                * **Blue (Attractive)**: Pulls the rocket towards its center.
                * **Red (Repulsive)**: Pushes the rocket away from its center.
                * These require the agent to exert more fuel to counteract their effects.
            * ⛽ **Fuel Stations (Green 'F')**: Refueling points that replenish the rocket's fuel. They are consumed upon use.
            * **Fuel Management**: Fuel is consumed with every step, and more rapidly when fighting gravity. Running out of fuel also terminates the episode.
            
            **Difficulty Levels**: Adjusting the difficulty changes the number of obstacles, gravity wells, fuel stations, obstacle speed, and maximum steps, thereby increasing the complexity of the navigation task.
            """)
        
        with tab3:
            st.markdown("""
            ### The Reinforcement Learning Agent:
            The core of this simulation is a **Proximal Policy Optimization (PPO)-enhanced Advantage Actor-Critic (A2C)** agent.
            
            * **Actor-Critic Architecture**: The agent consists of two neural networks:
                * **Actor (Policy Network)**: Decides which action to take (move left, right, up, down) based on the current state.
                * **Critic (Value Network)**: Estimates the "value" of being in a particular state, helping the actor make better decisions.
            * **PPO Enhancements**: PPO is an advanced policy gradient method that helps stabilize training by preventing the policy from changing too drastically in one update. This makes the learning process more robust and efficient.
            * **State Representation**: The agent observes 8 features: normalized X/Y position, normalized fuel, current gravity effect, normalized distance to nearest obstacle, normalized distance to nearest fuel station, and normalized X/Y distance to the goal.
            * **Reward System**: The agent learns through a system of rewards and penalties, encouraging goal-reaching, fuel efficiency, and obstacle avoidance.
            
            Adjust the **Learning Rate** and **Discount Factor (Gamma)** in the sidebar to observe their impact on the agent's learning behavior.
            """)

if __name__ == "__main__":
    main()
