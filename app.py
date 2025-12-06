import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, Response, jsonify
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import cv2  # OpenCV for image processing
import os

app = Flask(__name__)

# --- CONFIGURATION ---
# We check for all possible saved model names to find the best one available
POSSIBLE_MODELS = [
    "mspacman_vanilla_100k.pth",
    "mspacman_vanilla_50k.pth",
    "mspacman_cyclical_50k.pth",
    "mspacman_vanilla.pth",
    "mspacman_dqn.pth"
]

MODEL_PATH = None
for f in POSSIBLE_MODELS:
    if os.path.exists(f):
        MODEL_PATH = f
        break

if MODEL_PATH is None:
    print("WARNING: No model file found! The AI will play randomly.")

GAME_NAME = "ALE/MsPacman-v5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DQN ARCHITECTURE ---
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

# --- GLOBAL VARIABLES ---
env = None
policy_net = None
current_score = 0

def init_game():
    global env, policy_net
    # render_mode="rgb_array" allows us to grab the image without opening a window
    env = gym.make(GAME_NAME, frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)

    n_actions = env.action_space.n
    input_shape = (4, 84, 84)

    policy_net = DQN(input_shape, n_actions).to(device)
    
    if MODEL_PATH:
        try:
            print(f"Loading AI Model: {MODEL_PATH}")
            policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            policy_net.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

# Initialize environment on startup
init_game()

def generate_frames():
    """Video streaming generator function."""
    global env, policy_net, current_score
    
    state, _ = env.reset()
    current_score = 0
    done = False
    
    while True:
        if done:
            state, _ = env.reset()
            current_score = 0
            done = False

        # --- AI DECISION ---
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.uint8, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        # --- GAME STEP ---
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        current_score += reward

        # --- IMAGE PROCESSING ---
        # Get the full-color image from the game engine
        frame = env.unwrapped.render() 
        
        # Convert RGB (Gym) to BGR (OpenCV standards)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize for better visibility on the web page
        frame = cv2.resize(frame, (400, 500), interpolation=cv2.INTER_NEAREST)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Renders the HTML page."""
    return render_template('index.html', model_name=MODEL_PATH)

@app.route('/video_feed')
def video_feed():
    """Route for the video stream source."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_score')
def get_score():
    """API endpoint to update the score display."""
    global current_score
    return jsonify(score=int(current_score))

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)