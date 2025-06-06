import os
from datetime import datetime
import numpy as np
from time import sleep

print("Python script has started...")

# Get the current time and format it
current_time = datetime.now().strftime("%H:%M:%S")

# Create numpy array and calculate mean
my_array = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(my_array)

# Create the full path to the Desktop
desktop_path = os.path.expanduser("~/Desktop/hello.txt")

# Write the messages to the file
with open(desktop_path, "w") as file:
    file.write(f"Hello from {current_time}\n")
    file.write(f"the mean value is {mean_value}")

sleep(2)
print("Hello world!")