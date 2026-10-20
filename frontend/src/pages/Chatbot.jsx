import { useState } from "react";
import { askQuestion } from "../services/api";
import "./Chatbot.css";

function Chatbot() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!question.trim() || loading) {
      return;
    }

    const userQuestion = question.trim();

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: userQuestion,
      },
    ]);

    setQuestion("");
    setError("");
    setLoading(true);

    try {
      const data = await askQuestion(userQuestion, messages);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error);

      setError("Unable to get an answer. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-page">
      <div className="chatbot-container">
        <div className="chatbot-header">
          <h1>TrustInsure AI Assistant</h1>
          <p>
            Ask questions about health insurance terms and policies.
          </p>
        </div>

        <div className="chatbot-card">
          <div className="messages">
            {messages.length === 0 && (
              <p className="empty-message">
                Ask me anything about health insurance.
              </p>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.role}`}
              >
                <strong>
                  {message.role === "user"
                    ? "You"
                    : "AI Assistant"}
                </strong>

                <p>{message.content}</p>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <strong>AI Assistant</strong>
                <p>Thinking...</p>
              </div>
            )}
          </div>

          {error && (
            <p className="chat-error">
              {error}
            </p>
          )}

          <form
            className="chat-input-form"
            onSubmit={handleSubmit}
          >
            <input
              type="text"
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder="Ask a question..."
              disabled={loading}
            />

            <button
              type="submit"
              disabled={loading}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;