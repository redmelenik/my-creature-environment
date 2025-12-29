import React, { useState, useEffect, useCallback } from 'react';

const EvolutionVisualizer = () => {
  const [data, setData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const WORLD_SIZE = data?.summary.world_size || 100;
  const VIEWPORT_SIZE = 600; // Increased size for better view

  // --- 1. Data Fetching Logic ---
  const fetchState = useCallback(async () => {
    try {
      // Fetch the current state from the Flask server
      const response = await fetch('http://127.0.0.1:5000/api/state');
      const newState = await response.json();
      setData(newState);
      
      // Stop fetching if max generation (20) is reached or simulation is explicitly stopped
      if (newState.summary.generation >= 20 && !isRunning) setIsRunning(false); 
      
    } catch (error) {
      console.error("Could not fetch state. Is the Python server running on port 5000?", error);
    }
  }, [isRunning]);

  // --- 2. Polling Effect ---
  useEffect(() => {
    let interval;
    if (isRunning) {
      // Poll the backend every 2 seconds to get the new generation state
      interval = setInterval(fetchState, 2000); 
    }
    return () => clearInterval(interval);
  }, [isRunning, fetchState]);

  // --- 3. Start/Stop Handlers ---
  const handleStartStop = async () => {
    if (!isRunning) {
      // Tell the Python server to start the simulation thread
      await fetch('http://127.0.0.1:5000/api/start', { method: 'POST' });
      setIsRunning(true);
      fetchState(); // Fetch immediately after starting
    } else {
      // For simplicity, this just stops the frontend from refreshing.
      // A proper 'stop' endpoint would be needed in server.py to halt the thread.
      setIsRunning(false);
    }
  };

  // --- 4. Rendering Logic ---
  const renderCreatures = () => {
    if (!data || !data.creatures) return null;
    return data.creatures.map((c) => {
      const [x, y] = c.position;
      // Map world coordinates (0 to WORLD_SIZE) to SVG viewport pixels
      const px = (x / WORLD_SIZE) * VIEWPORT_SIZE;
      const py = (y / WORLD_SIZE) * VIEWPORT_SIZE;
      
      const fill = c.is_fittest ? 'red' : c.tribe_id ? 'blue' : 'gray';
      const size = c.alive ? 3 : 1; 
      
      return (
        <circle 
          key={c.id}
          cx={px} 
          cy={py} 
          r={size} 
          fill={fill}
          opacity={c.alive ? 1.0 : 0.3}
          title={`Fitness: ${c.fitness}, Tokens: B:${c.brain_tokens}`}
        />
      );
    });
  };

  return (
    <div style={{ padding: 20, fontFamily: 'Arial' }}>
      <h1>Evolutionary Simulation Viewer</h1>
      <button 
        onClick={handleStartStop}
        style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
      >
        {isRunning ? 'Stop Refresh' : 'Start Simulation'}
      </button>

      {data ? (
        <div style={{ display: 'flex', marginTop: 30, backgroundColor: '#f9f9f9', padding: 20, borderRadius: 8 }}>
          {/* Metrics Panel */}
          <div style={{ width: 250, marginRight: 40 }}>
            <h2>Generation {data.summary.generation}</h2>
            <p><strong>Fittest Score:</strong> {data.summary.fittest_score.toFixed(4)}</p>
            <p><strong>Alive:</strong> {data.summary.alive_count}/{data.creatures.length}</p>
            <p><strong>Fittest Tokens:</strong> B:{data.summary.fittest_tokens.B} | D:{data.summary.fittest_tokens.D} | L:{data.summary.fittest_tokens.L}</p>
            <p><strong>Status:</strong> {isRunning ? 'Running' : 'Paused / Complete'}</p>
          </div>

          {/* Visualization Area */}
          <svg
            width={VIEWPORT_SIZE}
            height={VIEWPORT_SIZE}
            style={{ border: '1px solid black', backgroundColor: 'white' }}
          >
            {/* Render Tokens (Food) */}
            {data.tokens.map((t, index) => {
              const [x, y] = t.position;
              const px = (x / WORLD_SIZE) * VIEWPORT_SIZE;
              const py = (y / WORLD_SIZE) * VIEWPORT_SIZE;
              
              let color = 'gold'; // Default
              if (t.type === 'brain') color = 'purple';
              if (t.type === 'body') color = 'orange';
              if (t.type === 'legs') color = 'green';

              return (
                <rect 
                  key={index}
                  x={px - 3} y={py - 3} width="6" height="6" fill={color}
                  title={t.type}
                />
              );
            })}

            {/* Render Creatures */}
            {renderCreatures()}
          </svg>
        </div>
      ) : (
        <p>Awaiting simulation start... Please run your Python server.</p>
      )}
    </div>
  );
};

export default EvolutionVisualizer;