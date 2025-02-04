import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved light directions from a CSV file
light_directions_2d = pd.read_csv('output/coin1/mlic/coin1_light_directions_2d.csv')

# Create a figure with a specified size
plt.figure(figsize=(12, 6))

# Scatter plot for 2D light directions
plt.subplot(1, 2, 1)  
plt.scatter(
    light_directions_2d['x'], 
    light_directions_2d['y'], 
    c='b',                     
    alpha=0.7,                 
    label="Light Directions"   
)

# Add horizontal and vertical lines at zero
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# Set title and labels for the scatter plot
plt.title("2D Light Directions Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()  
plt.grid(alpha=0.3) 

# Heatmap for light directions using seaborn
plt.subplot(1, 2, 2) 
sns.kdeplot(
    x=light_directions_2d['x'], 
    y=light_directions_2d['y'], 
    cmap='Blues',               
    fill=True,                  
    bw_method=0.2               
)

# Set title and labels for the heatmap
plt.title("2D Light Directions Heatmap")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(alpha=0.3) 

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()