import Prediction from "./pages/Prediction";
import Chatbot from "./pages/Chatbot";
import "./App.css";

function App() {
  return (
    <div className="app">
      {/* <header className="app-header">
        <h1>TrustInsure AI</h1>
        <p>Insurance Premium Prediction & AI Assistant</p>
      </header> */}

      <main className="dashboard">
        <section className="prediction-section">
          <Prediction />
        </section>

        <section className="chatbot-section">
          <Chatbot />
        </section>
      </main>
    </div>
  );
}

export default App;