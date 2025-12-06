import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import csv 
from collections import deque
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# --- Hyperparameters ---
GAME_NAME = "ALE/MsPacman-v5" 
BATCH_SIZE = 32
GAMMA           = 0.99
EPS_START       = 1.00   
EPS_END         = 0.05
TARGET_UPDATE   = 1000
LR              = 0.0001
MEMORY_SIZE     = 100000 
MIN_REPLAY_SIZE = 1000
SAVE_FILE       = "mspacman_cyclical_50k.pth" # Updated filename for this experiment
LOG_FILE        = "training_log_cyclical.csv"

# --- CYCLICAL SETTINGS (NEW) ---
NUM_EPISODES    = 50000
CYCLE_LENGTH    = 10000 # Reset Epsilon every 10k episodes
DECAY_FRACTION  = 0.8   # Spend 80% of the cycle decaying, 20% flat at EPS_END

# --- OLD SETTINGS (For Reference) ---
# EPS_DECAY = 500000 
# EPS_DROP  = 0.00038

# --- CHECKPOINT SETTINGS ---
CHECKPOINT_LIST = [1, 500, 1000, 10000, 25000, 50000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = convw
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def visualize_episode(policy_net, episode_num):
    print(f"\n>>> STARTING VISUALIZATION for Episode {episode_num} <<<")
    viz_env = gym.make(GAME_NAME, frameskip=1, render_mode="human")
    viz_env = AtariPreprocessing(viz_env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    viz_env = FrameStackObservation(viz_env, stack_size=4)
    
    state, _ = viz_env.reset()
    done = False
    viz_score = 0
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.uint8, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            
        state, reward, terminated, truncated, _ = viz_env.step(action)
        done = terminated or truncated
        viz_score += reward

    viz_env.close()
    print(f">>> VISUALIZATION COMPLETE. Score: {viz_score}\n")

def train():
    env = gym.make(GAME_NAME, frameskip=1, render_mode=None) 
    # Standard DeepMind Wrappers (Survival is key)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStackObservation(env, stack_size=4)

    n_actions = env.action_space.n
    input_shape = (4, 84, 84)

    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    
    # Check for existing save
    if os.path.exists(SAVE_FILE):
        print(f"Found {SAVE_FILE}. Resuming...")
        try:
            policy_net.load_state_dict(torch.load(SAVE_FILE))
        except:
            print("Starting fresh (Cyclical).")
    else:
        print(f"Starting FRESH Cyclical Training ({SAVE_FILE}).")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    # Initialize CSV logging
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Reward_Clipped', 'Real_Score', 'Epsilon', 'Steps'])

    print(f"Starting Cyclical DQN Training ({NUM_EPISODES} Episodes, Cycle={CYCLE_LENGTH})...")

    # Range starts at 1 so checkpoint "1" works correctly
    for i_episode in range(1, NUM_EPISODES + 1):
        
        # --- [OLD METHOD] STANDARD LINEAR DECAY (COMMENTED OUT) ---
        # Used for simple runs where we want to stabilize once and never explore again.
        # epsilon = max(EPS_END, EPS_START - (i_episode * EPS_DROP))
        
        # --- [NEW METHOD] CYCLICAL EPSILON (SAWTOOTH) ---
        # Logic: Reset Epsilon every 10k episodes to break out of "Local Optima" (stuck strategies).
        
        # 1. Determine position in current cycle (0 to 9999)
        cycle_idx = (i_episode - 1) % CYCLE_LENGTH
        
        # 2. Calculate Decay Rate: Drop from Start to End over the fraction of the cycle
        decay_steps = CYCLE_LENGTH * DECAY_FRACTION
        epsilon_drop = (EPS_START - EPS_END) / decay_steps
        
        # 3. Apply Logic
        if cycle_idx < decay_steps:
            epsilon = max(EPS_END, EPS_START - (cycle_idx * epsilon_drop))
        else:
            epsilon = EPS_END # Stay at floor for the rest of the cycle
        # --------------------------------------------------------
        
        state, _ = env.reset()
        total_reward = 0
        real_score = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.uint8, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            real_score += reward
            
            # --- STANDARD REWARD CLIPPING ---
            clipped_reward = np.sign(reward) 
            
            done = terminated or truncated
            total_reward += clipped_reward

            memory.push(state, action, clipped_reward, next_state, done)

            state = next_state
            steps_done += 1 

            if len(memory) > MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states_t = torch.tensor(states, dtype=torch.uint8, device=device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.uint8, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                current_q = policy_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                    expected_q = rewards_t + (1 - dones_t) * GAMMA * next_q

                loss = nn.MSELoss()(current_q.squeeze(), expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Log to Terminal
        print(f"Episode {i_episode} | Reward: {total_reward:.1f} | Real: {int(real_score)} | Ep: {epsilon:.4f}")

        # Log to CSV
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i_episode, total_reward, int(real_score), epsilon, steps_done])

        # Save Checkpoint based on list
        if i_episode in CHECKPOINT_LIST:
            checkpoint_name = f"mspacman_episode_{i_episode}.pth"
            torch.save(policy_net.state_dict(), checkpoint_name)
            print(f">>> SAVED CHECKPOINT: {checkpoint_name}")

        # Regular Save
        if i_episode % 100 == 0:
            torch.save(policy_net.state_dict(), SAVE_FILE)

    print("Training Complete.")
    torch.save(policy_net.state_dict(), SAVE_FILE)

if __name__ == "__main__":
    train()