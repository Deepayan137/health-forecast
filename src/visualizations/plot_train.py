import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
csv_path = "RealSimulatedData/training_1.csv"
df = pd.read_csv(csv_path, delimiter=';')

# Initialize the plot
plt.figure(figsize=(12, 6))

# Define color for each group
colors = {'Group1': 'red', 'Group2': 'blue'}

# Iterate over each unique ID and plot their trajectories
for id in df['Id'].unique():
    # Extract the subset of the DataFrame corresponding to the current ID
    subset = df[df['Id'] == id]
    
    # Extract the group of the current ID
    group = subset['Group'].unique()[0]
    
    # Plot the trajectory
    plt.plot(subset['Time'], subset['log10VL'], color=colors[group], alpha=0.5)

# Add labels and title
plt.xlabel('Time (Days)')
plt.ylabel('log10(Viral Load)')
plt.title('True Trajectories of Viral Load Over Time')

# Add legend
plt.legend(colors.keys())

# Show the plot
plt.show()
plt.savefig('final/training_trajectory.png')
