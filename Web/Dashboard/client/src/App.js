import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

function App() {
  // --- STATE ---
  const [inputs, setInputs] = useState({
    battery: 100,
    velocity: 0.5,
    temp: 40,
    latencyManual: 500
  });

  const [systemData, setSystemData] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showLogic, setShowLogic] = useState(false); 
  const [scenario, setScenario] = useState("Manual Control"); // Track active scenario

  // --- SCENARIO HANDLER ---
  const handleScenarioChange = (e) => {
    const selected = e.target.value;
    setScenario(selected);

    let newInputs = { ...inputs };

    switch (selected) {
      case "Start of Mission (Hover)":
        newInputs = { ...inputs, battery: 100, velocity: 0.5, temp: 40 };
        break;
      case "High Speed Survey":
        newInputs = { ...inputs, battery: 85, velocity: 8.0, temp: 65 };
        break;
      case "Thermal Throttling":
        newInputs = { ...inputs, battery: 60, velocity: 2.0, temp: 85 };
        break;
      case "Low Battery Return":
        newInputs = { ...inputs, battery: 15, velocity: 4.0, temp: 60 };
        break;
      case "Critical Survival":
        newInputs = { ...inputs, battery: 5, velocity: 1.0, temp: 70 };
        break;
      case "Manual Control":
      default:
        // Manual control doesn't force values, it just lets you edit them.
        // You could reset to defaults here if you wanted, but usually manual means "leave as is"
        break;
    }
    setInputs(newInputs);
  };

  // --- 1. TALK TO NODE.JS BRAIN (Debounced) ---
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/rdd-status', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputs),
        });
        const data = await response.json();
        setSystemData(data);
      } catch (error) {
        console.error("Connection Failed:", error);
      }
    };
    
    const timer = setTimeout(() => { fetchConfig(); }, 200);
    return () => clearTimeout(timer);
  }, [inputs]);

  // --- 2. PYTHON INFERENCE FUNCTION ---
  const runInference = useCallback(async (file, config) => {
    if (!file || !config) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mask', config.mask.join(','));
    formData.append('channels', config.channels);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setResultImage(imageUrl);
      }
    } catch (error) {
      console.error("Inference Error:", error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // --- 3. AUTO-UPDATE TRIGGER ---
  useEffect(() => {
    if (selectedFile && systemData && systemData.config) {
      runInference(selectedFile, systemData.config);
    }
  }, [systemData, selectedFile, runInference]);

  // --- HELPERS ---
  const handleSliderChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: Number(e.target.value) });
    setScenario("Manual Control"); // Switch back to manual if user touches a slider
  };

  const handleFileChange = (e) => {
    if(e.target.files[0]) {
        setSelectedFile(e.target.files[0]);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setResultImage(null);
    setIsProcessing(false);
  };

  const getStatusClass = (color) => {
    if (color === 'red') return 'text-red';
    if (color === 'orange') return 'text-orange';
    return 'text-green';
  };

  const calculatePhysicsLatency = () => {
    if (inputs.velocity > 0.1) return Math.floor(300 / inputs.velocity);
    return 1000;
  };

  return (
    <div className="container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <h2>Mission Control</h2>
        
        {/* NEW SCENARIO DROPDOWN */}
        <div className="control-group">
          <label>Quick Scenarios</label>
          <select 
            value={scenario} 
            onChange={handleScenarioChange} 
            className="scenario-select"
          >
            <option>Manual Control</option>
            <option>Start of Mission (Hover)</option>
            <option>High Speed Survey</option>
            <option>Thermal Throttling</option>
            <option>Low Battery Return</option>
            <option>Critical Survival</option>
          </select>
        </div>

        <hr style={{borderColor: '#444', marginBottom: '20px'}}/>

        <div className="control-group">
          <label>🔋 Battery Level <span>{inputs.battery}%</span></label>
          <input type="range" name="battery" min="0" max="100" value={inputs.battery} onChange={handleSliderChange} />
        </div>

        <div className="control-group">
          <label>🚀 Velocity <span>{inputs.velocity} m/s</span></label>
          <input type="range" name="velocity" min="0" max="15" step="0.5" value={inputs.velocity} onChange={handleSliderChange} />
        </div>

        <div className="control-group">
          <label>🔥 GPU Temp <span>{inputs.temp}°C</span></label>
          <input type="range" name="temp" min="30" max="95" value={inputs.temp} onChange={handleSliderChange} />
        </div>

        <div className="control-group">
          <label>⚙️ Latency Cap <span>{inputs.latencyManual} ms</span></label>
          <input type="range" name="latencyManual" min="50" max="1000" step="10" value={inputs.latencyManual} onChange={handleSliderChange} />
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main">
        <div className="header">
          <h1>Self-Aware RDD System</h1>
        </div>

        {systemData && systemData.config ? (
          <>
            {/* ROW 1: STATUS */}
            <div className="grid">
              <div className="card">
                <h4>System Mode</h4>
                <div className={`value ${getStatusClass(systemData.color)}`}>{systemData.mode}</div>
              </div>
              <div className="card">
                <h4>Active Config</h4>
                <div className="value">{systemData.config.name}</div>
              </div>
              <div className="card">
                <h4>Energy Cost</h4>
                <div className="value">{systemData.config.gflops} GFLOPs</div>
              </div>
            </div>

            {/* ROW 2: PHYSICS */}
            <div className="grid" style={{ marginTop: '20px' }}>
              <div className="card">
                <h4>GPU Temp</h4>
                <div className={inputs.temp > 80 ? "value text-red" : "value"}>
                  {inputs.temp}°C
                  {inputs.temp > 80 && <span className="text-small blink"> 🔥 OVERHEAT</span>}
                </div>
              </div>
              
              <div className="card">
                <h4>Latency Budget</h4>
                <div className="value">&lt; {Math.floor(systemData.effective_lat)} ms</div>
                <div className="text-small text-gray">Strictest limit applied</div>
              </div>

              <div className="card">
                <h4>Actual Latency</h4>
                <div className="value">~{systemData.config.latency_ms} ms</div>
                {/* MARGIN */}
                {(() => {
                  const margin = systemData.effective_lat - systemData.config.latency_ms;
                  const isSafe = margin >= 0;
                  return (
                    <div className={`text-small ${isSafe ? 'text-green' : 'text-red'}`}>
                      {isSafe ? '↑' : '↓'} {margin.toFixed(1)} ms margin
                    </div>
                  );
                })()}
              </div>
            </div>

            <hr className="divider"/>

            {/* LIVE FEED */}
            <h3>Live Drone Feed</h3>
            <div className="image-section">
              <div className="img-box relative-container">
                {selectedFile ? (
                   <>
                     <img src={URL.createObjectURL(selectedFile)} alt="Input" />
                     <button className="close-btn" onClick={clearImage}>×</button>
                   </>
                ) : (
                   <div className="placeholder">
                     <p>Select an Image</p>
                     <input type="file" onChange={handleFileChange} />
                   </div>
                )}
                <p>Original Input</p>
              </div>
              
              <div className="img-box">
                {isProcessing ? (
                  <div className="placeholder">Processing...</div>
                ) : resultImage ? (
                  <img src={resultImage} alt="Output" />
                ) : (
                  <div className="placeholder">Waiting for input...</div>
                )}
                <p>Real-Time Inference ({systemData.config.name})</p>
              </div>
            </div>

            {/* DECISION LOGIC */}
            <div className="logic-panel">
              <div className="logic-header" onClick={() => setShowLogic(!showLogic)}>
                <h3>System Decision Logic</h3>
                <span>{showLogic ? '▲' : '▼'}</span>
              </div>
              
              {showLogic && (
                <div className="logic-content">
                  <div className="logic-col">
                    <h4>1. Sensor Inputs</h4>
                    <p><strong>Battery:</strong> {inputs.battery}%</p>
                    <p><strong>Velocity:</strong> {inputs.velocity} m/s</p>
                    <p><strong>GPU Temp:</strong> {inputs.temp}°C</p>
                    <p><strong>Manual Latency:</strong> {inputs.latencyManual} ms</p>
                  </div>
                  
                  <div className="logic-col">
                    <h4>2. Derived Constraints</h4>
                    <p>
                        <strong>Physics Limit:</strong> {calculatePhysicsLatency()} ms
                        <br/>
                        <span className="text-small text-gray">(300 / Velocity)</span>
                    </p>
                    
                    {calculatePhysicsLatency() < inputs.latencyManual && (
                        <p className="text-orange">⚠️ Velocity overrides Manual Cap</p>
                    )}

                    <p>
                        <strong>Throttling State:</strong> <span className={inputs.temp >= 80 ? "text-red" : "text-green"}>
                            {inputs.temp >= 80 ? "ACTIVE (Protect Hardware)" : "Normal"}
                        </span>
                    </p>
                  </div>
                </div>
              )}
            </div>

          </>
        ) : (
          <h2 className="text-red">⚠️ SYSTEM FAILURE: BATTERY DEPLETED</h2>
        )}
      </div>
    </div>
  );
}

export default App;