import { useState } from 'react';
import type { ChangeEvent, SyntheticEvent } from 'react';


interface PredictionResponse {
  station_id: number;
  selected_time: string;
  bike_capacity: number;
  historical_available_bikes: number;
  predicted_available_bikes: number;
}

function App() {
  const [formData, setFormData] = useState({
    STATION_ID: 42,
    TARGET_TIME: "2021-11-15T12:00" 
  })

  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string>("")

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setFormData({ ...formData, [e.target.name]: value })
  }

  const handleSubmit = async (e: SyntheticEvent) => {
    e.preventDefault()
    setError("")
    setResult(null)
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || "API request failed")
      
      setResult(data)
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError("Failed to fetch prediction.")
      }
    }
  }

  return (
    <div className="container">
      <h1 className="title">Dublin Bikes Predictor</h1>
      <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '2rem', fontSize: '0.95rem', lineHeight: '1.5' }}>
        An end-to-end Machine Learning pipeline. Select a Dublin station ID and a target time <strong style={{color: 'var(--text-main)'}}>(Nov 1st - Nov 30th, 2021)</strong> to simulate a real-time availability prediction using historical data and XGBoost.
      </p>
      
      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Station ID</label>
            <input 
              type="number" 
              name="STATION_ID" 
              value={formData.STATION_ID} 
              onChange={handleChange} 
            />
          </div>

          <div className="form-group">
            <label>Target Time</label>
            <input 
              type="datetime-local" 
              name="TARGET_TIME" 
              value={formData.TARGET_TIME} 
              onChange={handleChange}
              min="2021-11-01T00:00"
              max="2021-11-30T23:59"
            />
          </div>

          <button type="submit" className="btn">
            Predict Availability
          </button>
        </form>
      </div>

      {error && <p style={{ color: '#ef4444', textAlign: 'center', marginTop: '1rem', fontWeight: 600 }}>{error}</p>}
      
      {result && (
        <div className="result-card">
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--result-accent)', marginBottom: '1rem', textAlign: 'center' }}>
            Station {result.station_id} Forecast
          </h2>
          
          <div className="result-row">
            <span>Selected Time</span>
            <strong>{new Date(result.selected_time).toLocaleString()}</strong>
          </div>
          <div className="result-row">
            <span>Station Capacity</span>
            <strong>{result.bike_capacity} bikes</strong>
          </div>
          <div className="result-row">
            <span>Current Available</span>
            <strong>{result.historical_available_bikes} bikes</strong>
          </div>
          
          <div className="result-main">
            <span>Predicted Available</span>
            <span style={{ fontSize: '1.5rem', color: '#fff' }}>{result.predicted_available_bikes.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default App