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

def run_simulation_thread():
    """Runs the entire simulation loop in a background thread."""
    global current_state, is_running
    is_running = True
    
    num_generations = 20 # Or whatever number you want
    
    for gen in range(env.current_generation, env.current_generation + num_generations):
        print(f"Server starting Generation {gen}...")
        
        # This is where your core simulation runs
        best_fitness = env.advance_generation(time_steps=200)
        
        # After each generation, update the global state
        current_state = env.export_state()
        time.sleep(1) # Give the frontend time to refresh

        if best_fitness >= 1.0:
            print("\n!!! GOAL ACHIEVED: Simulation stopping. !!!")
            break
            
    is_running = False
    print("Simulation thread finished.")

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