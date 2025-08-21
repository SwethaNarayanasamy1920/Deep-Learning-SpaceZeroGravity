# 🚀 Deep Space Explorer – Reinforcement Learning in Zero Gravity

An interactive **Streamlit + PyTorch** project that demonstrates **Reinforcement Learning (RL)** for dynamic space navigation in simulated **zero-gravity environments**.  

The agent (🚀) must learn to **navigate to a goal**, while avoiding obstacles, managing fuel, and dealing with **attractive and repulsive gravity wells**.  

---

## 🌌 Features
- **Custom Gym Environment**: Dynamic space grid with:
  - 🚀 Rocket agent (your RL learner)  
  - 🎯 Goal (mission target)  
  - 🔴 Moving obstacles  
  - 🕳️ Gravity wells (attractive & repulsive)  
  - ⛽ Fuel stations  
- **RL Agent**: PPO-enhanced **Advantage Actor-Critic (A2C)** implementation  
- **Interactive Training**: Tune hyperparameters in real-time via Streamlit UI  
- **Visualization**: Live environment rendering + performance charts  
- **Difficulty Modes**: Easy / Medium / Hard for different challenges  

---

## 📊 Demo / Live App
You can run the interactive demo locally or deploy it:  

### 🔹 Run Locally
```bash
# Clone repository
git clone https://github.com/SwethaNarayanasamy1920/Deep-Learning-SpaceZeroGravity.git
cd Deep-Learning-SpaceZeroGravity

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run RL.py
