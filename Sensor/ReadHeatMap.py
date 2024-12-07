import serial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure serial port (adjust these parameters to your setup)
SERIAL_PORT = 'COM10'  # Replace with your serial port
BAUD_RATE = 115200               # Replace with your baud rate

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# Set the number of columns and rows for your heatmap (32x24 matrix)
NUM_COLUMNS = 32  
NUM_ROWS = 24    

# Create an empty list to store the data
data = []

# Read data from serial port
while len(data) < NUM_COLUMNS * NUM_ROWS:
    # Read a line from the serial port (expecting values separated by spaces or tabs)
    line = ser.readline().decode('utf-8').strip()
    
    # Filter out any non-numeric values and split the line into individual temperature values
    values = line.split()  # Split by space or tab
    for value in values:
        try:
            # Try to convert the value to float, if it succeeds, add it to the data list
            data.append(float(value))
        except ValueError:
            # Skip any values that can't be converted to float (e.g., "Temperature")
            continue

    # If we have enough data, stop reading (in real applications, this would loop continuously)
    if len(data) >= NUM_COLUMNS * NUM_ROWS:
        break

# Convert the data into a numpy array and reshape it into a 32x24 matrix
data_matrix = np.array(data[:NUM_COLUMNS * NUM_ROWS]).reshape(NUM_ROWS, NUM_COLUMNS)


# Plot the heatmap using plt.imshow

plt.imshow(data_matrix, cmap='hot', interpolation='nearest')

# Add a color bar with label
plt.colorbar(label='Temperature (°C)')

# Add title and labels
plt.title('Infrared Sensor Heatmap')
plt.xlabel('Width (Pixels)')
plt.ylabel('Height (Pixels)')

# Display the heatmap
plt.show()


# # Create the heatmap
# plt.figure(figsize=(10, 8))  # Adjust the size of the plot
# sns.heatmap(data_matrix, annot=True, cmap='YlGnBu', cbar=True, fmt='.1f')

# # Add labels and title
# plt.title('Temperature Heatmap')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')

# # Show the heatmap
# plt.show()

# Close the serial connection
ser.close()