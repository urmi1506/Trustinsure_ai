import { useState } from "react";
import { predictPremium } from "../services/api";
import "./Prediction.css";

function Prediction() {
  const [formData, setFormData] = useState({
    age: "",
    sex: "male",
    bmi: "",
    children: 0,
    smoker: "no",
    region: "northeast",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setPrediction(null);
    setError("");

    try {
      const result = await predictPremium(formData);
      setPrediction(result);
    } catch (error) {
      console.error(error);
      setError("Unable to get prediction. Please check the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-page">
      <div className="prediction-container">

        <div className="prediction-header">
          <h1>TrustInsure AI</h1>
          <p>Health Insurance Premium Prediction</p>
        </div>

        <div className="prediction-card">
          <form
            className="prediction-form"
            onSubmit={handleSubmit}
          >
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                min="1"
                max="100"
                required
              />
            </div>

            <div className="form-group">
              <label>Sex</label>
              <select
                name="sex"
                value={formData.sex}
                onChange={handleChange}
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>

            <div className="form-group">
              <label>BMI</label>
              <input
                type="number"
                step="0.1"
                name="bmi"
                value={formData.bmi}
                onChange={handleChange}
                min="1"
                max="100"
                required
              />
            </div>

            <div className="form-group">
              <label>Children</label>
              <input
                type="number"
                name="children"
                min="0"
                max="10"
                value={formData.children}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label>Smoker</label>
              <select
                name="smoker"
                value={formData.smoker}
                onChange={handleChange}
              >
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Region</label>
              <select
                name="region"
                value={formData.region}
                onChange={handleChange}
              >
                <option value="northeast">Northeast</option>
                <option value="northwest">Northwest</option>
                <option value="southeast">Southeast</option>
                <option value="southwest">Southwest</option>
              </select>
            </div>

            <button
              className="predict-button"
              type="submit"
              disabled={loading}
            >
              {loading ? "Predicting..." : "Predict Premium"}
            </button>
          </form>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {prediction && (
            <div className="result-card">
              <h2>Estimated Annual Premium</h2>

              <p className="premium">
                ₹{prediction.predicted_premium.toFixed(2)}
              </p>

              <h3>Premium Explanation</h3>

              <div className="explanation-list">
                {prediction.explanation.map((item) => (
                  <div
                    className="explanation-item"
                    key={item.feature}
                  >
                    <span>{item.feature}</span>

                    <span>
                      {item.impact >= 0 ? "+" : ""}
                      {item.impact.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default Prediction;
