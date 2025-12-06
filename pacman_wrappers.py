import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    Implements "Hardcore Survival" reward shaping.
    
    Changes:
    1. Death Penalty (-300): Extreme penalty. Survival is the #1 priority.
    2. Pellet Reward (+2): High incentive to clear the board.
    3. Event Bonus (+10/+20): High reward for Fruit/Ghost Combos.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        info['raw_reward'] = raw_reward # Track real score

        custom_reward = 0.0
        current_lives = self.env.unwrapped.ale.lives()
        
        # --- 1. DEATH PENALTY (HARDCORE) ---
        # -300 points. The agent must prioritize staying alive above all else.
        if current_lives < self.lives:
            custom_reward -= 100.0 
        
        # --- 2. AGGRESSIVE POSITIVE REINFORCEMENT ---
        if raw_reward > 0:
            # Pellet (Score < 50)
            # Reward: +2.0. Strong incentive to clear pellets.
            if raw_reward < 50:
                custom_reward += 5.0 
            
            # Fruit / Single Ghost (Score 50-500)
            # Reward: +10.0.
            elif 50 <= raw_reward <= 500:
                custom_reward += 1.0
            
        
        self.lives = current_lives
        return obs, custom_reward, terminated, truncated, info