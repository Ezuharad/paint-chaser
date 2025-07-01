import matplotlib.pyplot as plt

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from paint_chaser.model import ResidualAgent
from paint_chaser.environment.environment import UFO50PaintChaseEnv

# Training hyperparameters
LEARNING_RATE = 1e-3  # 1e-3
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 999999999999
GAMMA = 0.995  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Initialize environment and agent
env = UFO50PaintChaseEnv()

agent = ResidualAgent()
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

# Training metrics
episode_rewards = []
epsilon = EPSILON_START

reward_file = open("reward.log", "w")

# Training loop
for episode in tqdm(range(NUM_EPISODES), desc="Training"):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Epsilon-greedy action selection
        if torch.rand(1) < epsilon:
            action = torch.randint(0, 4, (1,)).item()
        else:
            with torch.no_grad():
                q_values = agent(state.unsqueeze(0))
                action = q_values.argmax().item()

        # Take action and observe next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward.item()

        # Store transition in memory (for future implementation of replay buffer)
        # For now, we'll do immediate learning

        # Calculate TD target and loss
        with torch.no_grad():
            next_q_values = agent(next_state.unsqueeze(0))
            next_q_value = next_q_values.max()
            td_target = reward + GAMMA * next_q_value * (not terminated)

        # Calculate current Q value and loss
        current_q_values = agent(state.unsqueeze(0))
        current_q_value = current_q_values[0, action]
        loss = nn.MSELoss()(current_q_value, td_target)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if terminated or truncated:
            break

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Record metrics
    episode_rewards.append(episode_reward)
    reward_file.write(f"{episode_reward}\n")

    # Print progress every 10 episodes
    if (episode + 1) % 5 == 0:
        print(f"Episode: {episode}")
        print(f"{sum(episode_rewards[-10:]) / 10}\n")
        torch.save(agent.state_dict(), f"model_latest.pt")

torch.save(agent.state_dict(), f"model_{episode}.pt")

print("Training completed!")
