import os
from datetime import datetime

# Get the current time and format it
current_time = datetime.now().strftime("%H:%M:%S")

# Create the full path to the Desktop
desktop_path = os.path.expanduser("~/Desktop/hello.txt")

# Write the message to the file
with open(desktop_path, "w") as file:
    file.write(f"Hello from {current_time}")