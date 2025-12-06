import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURATION ---
# Check which log file exists.
LOG_FILE = 'training_log_cyclical.csv' 
WINDOW_SIZE = 100 

print(f"Looking for {LOG_FILE}...")

if not os.path.exists(LOG_FILE):
    if os.path.exists('training_log_50k.csv'):
        print(f"Found 'training_log_50k.csv'. Using that.")
        LOG_FILE = 'training_log_50k.csv'
    elif os.path.exists('training_log.csv'):
        print(f"Found 'training_log.csv'. Using that.")
        LOG_FILE = 'training_log.csv'
    else:
        print(f"ERROR: Could not find any CSV file. Make sure training started!")
        exit()

# 1. Load Data
try:
    df = pd.read_csv(LOG_FILE)
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Set the visual style
sns.set_theme(style="whitegrid")

# --- GRAPH 1: The Standard Learning Curve (Line Plot) ---
plt.figure(figsize=(12, 6))
plt.title(f"Agent Performance (Rolling Average of {WINDOW_SIZE})", fontsize=16)
plt.xlabel("Episode")
plt.ylabel("Real Game Score")

# Plot raw data as faint background noise
plt.plot(df['Episode'], df['Real_Score'], color='gray', alpha=0.1, label='Raw Score')

# Calculate and plot Moving Average
df['Rolling_Score'] = df['Real_Score'].rolling(window=WINDOW_SIZE).mean()
sns.lineplot(data=df, x='Episode', y='Rolling_Score', color='#FFD700', linewidth=2.5, label=f'{WINDOW_SIZE}-Episode Moving Avg')

plt.legend()
plt.tight_layout()
plt.savefig("graph_1_learning_curve.png")
print("Saved graph_1_learning_curve.png")
plt.close()

# --- GRAPH 2: Scatter Plot (Exploration vs. Performance) ---
plt.figure(figsize=(10, 8))
plt.title("Impact of Exploration (Epsilon) on Score", fontsize=16)

# Scatter plot: Epsilon on X, Score on Y
sc = plt.scatter(df['Epsilon'], df['Real_Score'], 
                 c=df['Episode'], cmap='viridis', alpha=0.3, s=10)

plt.colorbar(sc, label='Episode Number (Age of Agent)')
plt.gca().invert_xaxis() # Invert X so "Smart" (0.05) is on the right
plt.xlabel("Epsilon (Exploration Rate)")
plt.ylabel("Real Score")
plt.tight_layout()
plt.savefig("graph_2_epsilon_scatter.png")
print("Saved graph_2_epsilon_scatter.png")
plt.close()

# --- GRAPH 3: The Numerical Matrix Heatmap ---
# This visualizes EXACTLY how many games landed in each score bracket
plt.figure(figsize=(14, 8))
plt.title("Performance Matrix: Frequency of Scores per Training Phase", fontsize=16)

# 1. Create Bins for Episodes (Phases of Training)
# We divide the total run into 10 equal chunks (e.g. 0-5k, 5k-10k...)
num_bins = 10
df['Episode_Phase'] = pd.cut(df['Episode'], bins=num_bins)

# 2. Create Bins for Scores (Score Brackets)
# We define clear brackets: 0-250, 250-500, etc. up to the max score
max_score = df['Real_Score'].max()
# Ensure we have at least one bin if max_score is 0 (unlikely)
if max_score == 0: max_score = 100
score_bins = range(0, int(max_score) + 500, 250) # Buckets of 250 points
df['Score_Bracket'] = pd.cut(df['Real_Score'], bins=score_bins)

# 3. Create the Matrix (Pivot Table)
# This counts how many times a score occurred in a specific phase
matrix_data = pd.crosstab(df['Score_Bracket'], df['Episode_Phase'])

# 4. Plot Heatmap with Numbers (Annot=True)
# fmt='d' ensures numbers are integers (e.g., 342) not floats (342.0)
sns.heatmap(matrix_data, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Number of Games'})

# Clean up axes for readability
plt.gca().invert_yaxis() # Put high scores at the top
plt.ylabel("Score Range")
plt.xlabel("Training Phase (Episode Range)")
plt.tight_layout()
plt.savefig("graph_3_matrix_heatmap.png")
print("Saved graph_3_matrix_heatmap.png")
plt.close()

print("All graphs generated successfully!")