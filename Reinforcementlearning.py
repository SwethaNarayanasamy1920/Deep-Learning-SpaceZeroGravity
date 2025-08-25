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
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #4da6ff;
    }
    .stProgress > div > div {
        background-color: #4da6ff;
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
        
        # Add stars in the background
        num_stars = 200
        star_positions = np.random.rand(num_stars, 2) * self.grid_size
        star_sizes = np.random.exponential(0.7, num_stars)
        star_colors = np.random.choice(['white', '#FFFDD0', '#F8F7FF', '#CAE9FF'], num_stars)
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
            
            num_circles = 12
            for i in range(num_circles):
                radius = well['radius'] * (i+1)/num_circles
                alpha = 0.4 * (1 - i/num_circles)
                color_val = cmap(0.7 - 0.5 * i/num_circles)
                circle = plt.Circle(well['position'], radius, color=color_val, 
                                   alpha=alpha, fill=True, zorder=2)
                ax.add_patch(circle)
            
            ax.scatter(well['position'][0], well['position'][1], 
                      color='white', s=50, zorder=3, edgecolor=color)
            
            sign = "+" if well['strength'] > 0 else "−"
            text = ax.text(well['position'][0], well['position'][1], sign, 
                   ha='center', va='center', color='white', fontsize=12, 
                   fontweight='bold', zorder=4)
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Draw fuel stations
        for station in self.fuel_stations:
            glow = plt.Circle(station['position'], 1.0, color='green', alpha=0.2, zorder=3)
            station_marker = plt.Circle(station['position'], 0.5, color='lime', alpha=0.8, zorder=4)
            ax.add_patch(glow)
            ax.add_patch(station_marker)
            text = ax.text(station['position'][0], station['position'][1], 'F', ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=5)
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
        
        # Draw obstacles
        for obs in self.obstacles:
            obstacle = plt.Circle(obs['position'], obs['radius'], color='red', alpha=0.7, zorder=5)
            ax.add_patch(obstacle)
            danger_ring = plt.Circle(obs['position'], obs['radius'] + 0.3, color='red', alpha=0.2, fill=False, linestyle='--', zorder=5)
            ax.add_patch(danger_ring)
            if np.linalg.norm(obs['velocity']) > 0:
                ax.arrow(obs['position'][0], obs['position'][1], obs['velocity'][0]*3, obs['velocity'][1]*3, head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7, zorder=6)
        
        # Draw goal
        goal_glow = plt.Circle(self.goal, 1.5, color='green', alpha=0.15, zorder=5)
        goal_outer = plt.Circle(self.goal, 1.0, color='green', alpha=0.4, zorder=6)
        goal_marker = plt.Circle(self.goal, 0.8, color='#00FF00', alpha=0.8, zorder=6)
        ax.add_patch(goal_glow)
        ax.add_patch(goal_outer)
        ax.add_patch(goal_marker)
        text = ax.text(self.goal[0], self.goal[1], 'G', ha='center', va='center', color='white', fontsize=12, fontweight='bold', zorder=7)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='green')])
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color='cyan', alpha=0.5, linestyle=':', marker='.', markersize=4, zorder=4)
            
        # Draw rocket
        engine_glow = plt.Circle(self.rocket_pos, 0.8, color='#00FFFF', alpha=0.2, zorder=7)
        rocket_marker = plt.Circle(self.rocket_pos, 0.6, color='cyan', alpha=1.0, zorder=8)
        ax.add_patch(engine_glow)
        ax.add_patch(rocket_marker)
        
        # Set axis labels and title
        ax.set_title("Deep Space Explorer", color='white', fontsize=18)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')  # Hide the axes for a cleaner look
        
        # Return the figure object
        return fig
    
    def close(self):
        plt.close('all')

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

# Main Streamlit application function
def main():
    st.sidebar.header("Configuration")
    difficulty = st.sidebar.radio("Select Difficulty", ('easy', 'medium', 'hard'), index=1)
    
    num_episodes = st.sidebar.number_input("Number of Training Episodes", min_value=100, max_value=5000, value=1000, step=100)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=0.001, step=0.0001, format="%.5f")
    gamma = st.sidebar.number_input("Discount Factor (Gamma)", min_value=0.5, max_value=0.99, value=0.99, step=0.01)

    if "model" not in st.session_state:
        st.session_state.model = None

    if st.sidebar.button("Start Training"):
        st.session_state.model = train_agent(difficulty, num_episodes, learning_rate, gamma)
    
    if st.session_state.model and st.sidebar.button("Test Trained Agent"):
        test_agent(st.session_state.model, difficulty)

    # Add description and instructions
    with st.expander("About this application"):
        st.markdown("""
        ## Dynamic Space Exploration with Reinforcement Learning
        
        This application demonstrates reinforcement learning in a dynamic space environment. 
        The agent (rocket) must navigate to the goal while avoiding obstacles, managing fuel, 
        and dealing with gravitational effects.
        
        ### Environment Features:
        - Moving obstacles that must be avoided
        - Gravitational wells (blue: attractive, red: repulsive)
        - Fuel stations for refueling
        - Limited fuel supply
        
        ### Agent Capabilities:
        - Learns using a Proximal Policy Optimization (PPO) algorithm
        - Adapts to different difficulty levels
        - Learns to plan efficient paths considering fuel consumption
        
        ### How to use:
        1. Select difficulty level from the sidebar
        2. Configure agent parameters if desired
        3. Click "Start Training" to train the agent
        4. After training, click "Test Trained Agent" to see how it performs
        
        The visualization shows the rocket (cyan), goal (green), obstacles (red), 
        fuel stations (green F), and gravity wells (blue/red gradient).
        """)

# Training function
def train_agent(difficulty, num_episodes, learning_rate, gamma):
    env = DynamicSpaceExplorationEnv(difficulty=difficulty)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from the model
            probs, value = model(state_tensor)
            dist = distributions.Categorical(probs)
            action = dist.sample()
            
            # Step the environment
            next_state, reward, done, _ = env.step(action.item())
            
            # Store values for backpropagation
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            episode_reward += reward
        
        # Calculate returns and advantages
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        
        advantages = returns - values.squeeze()
        
        # PPO-like update (simplified)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_rewards.append(episode_reward)
        
        # Update progress and status
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        status_text.text(f"Training Episode: {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

    st.success("Training complete!")
    return model

# Testing function
def test_agent(model, difficulty):
    env = DynamicSpaceExplorationEnv(difficulty=difficulty)
    state = env.reset()
    done = False
    episode_reward = 0
    
    st.subheader("Testing Trained Agent")
    plot_placeholder = st.empty()
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from the model (greedy policy)
        probs, _ = model(state_tensor)
        action = torch.argmax(probs).item()
        
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        episode_reward += reward
        
        # Plot and display the current state
        fig = env.plot_state()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Step", env.current_step)
        with col2:
            st.metric("Fuel Remaining", f"{info['fuel']:.1f}")
        with col3:
            st.metric("Distance to Goal", f"{info['distance_to_goal']:.1f}")
        with col4:
            st.metric("Total Reward", f"{episode_reward:.2f}")
            
        time.sleep(0.05)
    
    st.success(f"Testing finished! Final reward: {episode_reward:.2f}")


if __name__ == "__main__":
    main()
