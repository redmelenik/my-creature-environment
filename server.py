from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
# Assuming your Environment class and dependencies are imported here
from evolutionary_sim import Environment # You may need to adjust this import

app = Flask(__name__)
CORS(app) # Allows React app running on a different port to access

# Initialize the environment once
env = Environment(world_size=100, num_creatures=100)
current_state = env.export_state()
is_running = False
sim_thread = None   

def run_generation_thread():
    """Runs a single generation's worth of steps in a background thread."""
    global current_state, is_running, env
    is_running = True
    
    print(f"Server starting Generation {env.current_generation}...")
    
    # Run the core simulation steps (e.g., 200 steps)
    best_fitness = env.advance_generation(time_steps=200)
    
    # After the generation is complete, update the global state
    current_state = env.export_state()
    
    if best_fitness >= 1.0:
        print("\n!!! GOAL ACHIEVED: Simulation stopping. !!!")
        
    is_running = False
    print(f"Generation {env.current_generation - 1} thread finished.")

@app.route('/api/state', methods=['GET'])

def get_state():
    """Endpoint for the React app to fetch the current world state."""
    return jsonify(current_state)

@app.route('/api/start', methods=['POST'])

def start_simulation():
    """Endpoint to start the simulation thread."""
    global is_running
    if not is_running:
        thread = threading.Thread(target=run_simulation_thread)
        thread.start()
        return jsonify({"status": "Simulation started"}), 200
    else:
        return jsonify({"status": "Simulation is already running"}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(port=5000, debug=False) # Run on port 5000