import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Define parameters
num_samples = 1000  # Adjust the number of records you need
start_time = datetime.now()
time_step = timedelta(seconds=1)  # 1-second intervals

# Generate synthetic drone sensor data
data = []
for i in range(num_samples):
    timestamp = start_time + i * time_step
    lidar = round(np.random.uniform(0.5, 5.0), 2)  # Simulated LiDAR distance (meters)
    ultrasonic = round(np.random.uniform(0.5, 5.0), 2)  # Simulated Ultrasonic distance
    imu_x = round(np.random.uniform(-1.0, 1.0), 2)  # Accelerometer X
    imu_y = round(np.random.uniform(-1.0, 1.0), 2)  # Accelerometer Y
    imu_z = round(np.random.uniform(9.5, 10.5), 2)  # Accelerometer Z (Gravity effect)
    obstacle = 1 if lidar < 2.0 or ultrasonic < 2.0 else 0  # If distance < 2m, assume obstacle

    data.append([timestamp, lidar, ultrasonic, imu_x, imu_y, imu_z, obstacle])

# Create DataFrame
df = pd.DataFrame(data, columns=['Timestamp', 'LiDAR', 'Ultrasonic', 'IMU_X', 'IMU_Y', 'IMU_Z', 'Obstacle'])

# Save to CSV
df.to_csv('drone_sensor_data.csv', index=False)

# Fix for Windows Unicode issue (removing emoji)
print("Drone sensor dataset generated: drone_sensor_data.csv")
