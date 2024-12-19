import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file using the absolute path
data = pd.read_excel(r'C:\Users\thaiv\OneDrive\Desktop\PILS\data\2024-12-10_12-23-04.xlsx', header=None)  # Use header=None if there's no header row

# Convert the data to a NumPy array
matrix = data.values

# Check the shape of the matrix
if matrix.shape != (32, 24):
    print(f"Warning: Matrix shape is {matrix.shape}, expected (32, 24).")

# Plot the heatmap
plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature (°C)')  # Add a color bar
plt.title('Infrared Sensor Heatmap')    # Add a title
plt.xlabel('Width (Pixels)')            # X-axis label
plt.ylabel('Height (Pixels)')           # Y-axis label

# Display the heatmap
plt.show()
