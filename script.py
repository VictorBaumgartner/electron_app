import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a FastAPI app instance
app = FastAPI()

# Add CORS middleware to allow cross-origin requests from the Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory data store
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1

    def reset(self):
        self.value = 0

    def get(self):
        return self.value

# Create a single counter instance
counter = Counter()

# A list to keep track of active WebSocket connections
active_connections: List[WebSocket] = []

async def notify_clients():
    """Notifies all connected clients of the current count."""
    if active_connections:
        message = str(counter.get())
        logger.info(f"Broadcasting count to {len(active_connections)} clients: {message}")
        # Create a list of tasks to send messages concurrently
        tasks = [conn.send_text(message) for conn in active_connections]
        await asyncio.gather(*tasks)

@app.post("/add")
async def add_one():
    """Endpoint to increment the count by one."""
    counter.increment()
    logger.info(f"Count incremented to: {counter.get()}")
    # Notify all clients about the change
    await notify_clients()
    return {"message": "Count updated", "current_count": counter.get()}

@app.post("/reset")
async def reset_count():
    """Endpoint to reset the count to zero."""
    counter.reset()
    logger.info("Count reset to 0")
    # Notify all clients about the change
    await notify_clients()
    return {"message": "Count reset", "current_count": counter.get()}

@app.get("/count")
async def get_count():
    """Endpoint to get the current count."""
    logger.info(f"Replying with current count: {counter.get()}")
    return {"current_count": counter.get()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live count updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("New client connected via WebSocket.")
    try:
        # Send the current count immediately on connection
        await websocket.send_text(str(counter.get()))
        while True:
            # Keep the connection alive to listen for disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# Note: This file will be run by uvicorn from main.js, so a __main__ block is not needed.
# For standalone testing, you would run: uvicorn script:app --reload