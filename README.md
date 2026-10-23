# TrustInsure AI
### Explainable Health Insurance Premium Prediction & RAG-Powered Insurance Assistant

TrustInsure AI is an AI-powered application designed to make health insurance information more accessible and understandable. It combines machine learning-based premium estimation, explainable AI, and Retrieval-Augmented Generation (RAG) to help users explore estimated insurance costs and understand insurance-related documents.

The project brings together **predictive machine learning, explainable AI, semantic search, and generative AI** in a single application.

## ✨ Key Features

- **Premium Prediction** — Estimates health insurance charges from user-provided demographic and lifestyle features.
- **Explainable AI** — Uses SHAP to explain how input features influence a model prediction.
- **RAG-Based Chatbot** — Answers insurance-related questions using information retrieved from insurance documents.
- **Hybrid Search** — Combines dense semantic retrieval with sparse keyword-based retrieval to improve document discovery.
- **Document Processing** — Extracts and processes PDF content into searchable chunks.
- **REST API** — Uses FastAPI to connect machine learning and RAG functionality to the frontend.
- **Interactive Web Interface** — Built with React for premium estimation and insurance question answering.

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, JavaScript, Vite |
| Backend | Python, FastAPI |
| Machine Learning | TensorFlow, Keras, Pandas, NumPy, Scikit-learn |
| Explainable AI | SHAP |
| Document Processing | Docling |
| Embeddings | Hugging Face Sentence Transformers |
| Vector Database | Qdrant |
| Retrieval | Dense and sparse hybrid search |
| Generative AI | Google Gemini API |
| Version Control | Git, GitHub |

## 🏗️ Architecture

                         User
                          |
                          v
                    React Frontend
                          |
                     REST API / JSON
                          |
                          v
                    FastAPI Backend
                     /           \
                    /             \
                   v               v
          Premium Prediction    RAG Assistant
                   |               |
             Input Validation   Insurance PDFs
                   |               |
             Preprocessing      Docling Extraction
                   |               |
             Trained DNN        Semantic Chunking
                   |               |
                   |          Dense + Sparse Search
                   |               |
                   |             Qdrant
                   |               |
                   |         Retrieved Context
                   |               |
                   |           Gemini API
                   |               |
                   v               v
             Premium Estimate   Generated Answer
                   |
                   v
              SHAP Explanation


## 🔍 How It Works

### 1. Health Insurance Premium Prediction

The prediction pipeline uses supervised regression to estimate continuous insurance charges.

1. Users provide details such as age, sex, BMI, number of children, smoking status, and region.
2. The React frontend sends the inputs to the FastAPI backend.
3. The backend validates the request and applies the fitted preprocessing pipeline.
4. The transformed features are passed to the trained neural network.
5. The model returns an estimated insurance charge.
6. SHAP provides an explanation of the model's prediction.

### 2. RAG-Powered Insurance Assistant

The chatbot retrieves relevant information from insurance documents before generating an answer.

1. PDF documents are processed using Docling.
2. Extracted content is prepared for retrieval.
3. Semantic chunking divides the content into meaningful sections.
4. Dense embeddings and sparse keyword representations support hybrid retrieval.
5. Qdrant stores and retrieves the indexed document chunks.
6. Relevant context is supplied to the Gemini model.
7. The model generates an answer grounded in the retrieved context.

The assistant is designed to acknowledge when the provided documents do not contain enough information to answer a question.

## 🚀 Getting Started

### Prerequisites

- Git
- Python 3.12 or a version supported by the project's dependencies
- Node.js and npm
- A running Qdrant instance
- A Google Gemini API key
- The trained model and its compatible preprocessing artifact

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Set Up the Backend

Navigate to the backend directory and create a virtual environment.

```bash
cd backend
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install backend dependencies if the project uses `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a local `.env` file based on the project's environment template.

```text
GEMINI_API_KEY=your_gemini_api_key
```

Never commit real API keys, credentials, or private environment files.

### 4. Start Qdrant

Start Qdrant and verify that it is accessible at the URL configured by the backend. The current local configuration uses:

`http://localhost:6333`

Ensure the required document collection has been indexed with the matching embedding configuration.

### 5. Run the Backend

Use the FastAPI entry point configured in your repository. For an application exposed as `src.api.main:app`, the command is:

```bash
uvicorn src.api.main:app --reload
```

The backend should be available at `http://localhost:8000`.

If enabled, API documentation is available at `http://localhost:8000/docs`.

### 6. Run the Frontend

Open another terminal and navigate to the frontend directory.

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically `http://localhost:5173`.

> Commands and directory paths may need adjustment to match the current repository structure.

## 🧪 Testing and Quality

The following checks are recommended before merging changes:

- Backend starts without errors.
- Valid prediction requests return finite numerical estimates.
- Invalid API inputs are handled correctly.
- Model and preprocessing artifacts remain compatible.
- SHAP explanations are generated successfully.
- Retrieval returns relevant chunks for representative insurance questions.
- The chatbot handles questions unsupported by the retrieved context.
- Frontend checks and production builds pass.
- Secrets are excluded from version control.

## 🤝 Contributing

**Contributions are welcome!** Whether you're a student, developer, AI/ML enthusiast, or open-source contributor, you can help improve TrustInsure AI.

You can contribute by:

- Fixing bugs or improving error handling
- Improving the React user interface and accessibility
- Writing backend or frontend tests
- Improving data preprocessing and model evaluation
- Exploring better retrieval, chunking, or ranking strategies
- Improving RAG answer quality and evaluation
- Improving documentation and setup instructions

### How to Contribute

1. **Fork** this repository.
2. **Clone** your fork to your local machine.
3. **Create a branch** for your changes.
4. Make your changes and test them locally.
5. Commit your changes with a clear, descriptive message.
6. Push your branch to your fork.
7. Open a **Pull Request (PR)** explaining what you changed, why it is useful, and how you tested it.

Please keep pull requests focused, avoid committing secrets or personal data, and include relevant tests where practical.

For significant changes, consider opening an issue first to discuss the proposed approach.

## 🔐 Privacy and Responsible Use

- Premium outputs are estimates, not official insurance quotations.
- Model explanations describe learned model behavior and do not establish causation.
- Chatbot responses depend on the available documents and retrieval quality.
- Avoid uploading real customer records or sensitive personal information to public issues or test fixtures.
- API keys and other credentials must be managed securely.

**Built with a focus on explainable machine learning, document-grounded AI, and accessible insurance information.**

If you find the project useful, consider starring the repository, sharing feedback, or contributing improvements!
