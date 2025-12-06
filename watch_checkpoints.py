import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import time
import os

# --- UPDATED LIST ---
# We added 40000 to the list.
CHECKPOINTS_TO_WATCH = [1, 500, 1000, 10000, 25000, 40000]
GAME_NAME = "ALE/MsPacman-v5"

# --- DQN CLASS ---
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

def watch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on: {device}")

    env = gym.make(GAME_NAME, frameskip=1, render_mode="human")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)

    n_actions = env.action_space.n
    input_shape = (4, 84, 84)
    policy_net = DQN(input_shape, n_actions).to(device)

    for episode_num in CHECKPOINTS_TO_WATCH:
        # Standard filename pattern
        filename = f"mspacman_episode_{episode_num}.pth"
        
        # --- SMART FALLBACK LOGIC ---
        # If the specific episode file is missing (like 40,000), 
        # check if it's the final run and grab the main save file instead.
        if not os.path.exists(filename):
            if episode_num == 40000:
                print(f"Checkpoint 40000 not found. Checking for main save file...")
                # Try the Cyclical name first, then Vanilla
                if os.path.exists("mspacman_cyclical_50k.pth"):
                    filename = "mspacman_cyclical_50k.pth"
                elif os.path.exists("mspacman_vanilla_50k.pth"):
                    filename = "mspacman_vanilla_50k.pth"
                elif os.path.exists("mspacman_dqn.pth"):
                    filename = "mspacman_dqn.pth"
            
        if not os.path.exists(filename):
            print(f"Skipping Episode {episode_num} (File not found).")
            continue

        print(f"\n>>> LOADING CHECKPOINT: {filename} <<<")
        print(f"Showing Episode {episode_num}...")
        time.sleep(2)
        
        try:
            policy_net.load_state_dict(torch.load(filename, map_location=device))
            policy_net.eval()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        state, _ = env.reset()
        done = False
        total_score = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), dtype=torch.uint8, device=device).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_score += reward
            
        print(f"Episode {episode_num} Finished. Score: {total_score}")
        time.sleep(2)

    env.close()

if __name__ == "__main__":
    watch()