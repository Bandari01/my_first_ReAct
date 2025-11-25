# AI Agent System For Data Analytics

An AI-powered data analysis system based on multiple agent architectures, capable of automatically solving Kaggle competition problems.

## 🎯 Overview

This project implements four AI agent architectures — **Direct LLM Prediction (DLP)**, **ReAct**, **RAG** and **Multi-Agent** — for end-to-end automated data analysis and Kaggle submission generation. By simply providing a Kaggle competition link and selecting an AI architecture, the system will automatically:

* Retrieve and analyze the dataset
* Generate and execute analytical/modeling code
* Export the `submission.csv`
* Record performance metrics and runtime logs

## 🏗️ System Architecture

### Architecture 1: DLP Agent (Direct LLM Prediction)

* Directly uses the LLM model to perform predictions on the data.
* Leverages the model's internal knowledge for prediction tasks.

### Architecture 2: ReAct Agent (Reasoning + Acting Loop)

* Uses LLM to generate machine learning code and automatically fixes errors.
* Iterative **think → act → observe** process suitable for exploratory tasks.

### Architecture 3: RAG Agent (Retrieval-Augmented Generation)

* Retrieves similar competition data and code snippets from the **knowledge_base/**.
* Uses data retrieval to generate better and more robust analytical code.

### Architecture 4: Multi-Agent System

* Uses LLM to first generate a comprehensive plan, then executes it step-by-step.
* Coordinates execution to complete complex tasks.

> Users can select the preferred agent type in the frontend; the system will automatically execute the corresponding workflow.

## 📁 Project Structure

```
ai-agent-analytics/
├── backend/
│   ├── agents/              # Agent implementations (DLP, ReAct, RAG, Multi-Agent)
│   ├── evaluation/          # Evaluation metrics and comparison tools
│   ├── executor/            # Code execution engine
│   ├── kaggle/              # Kaggle API integration
│   ├── llm/                 # LLM client wrapper
│   ├── RAG_tool/            # RAG specific tools and knowledge base
│   ├── utils/               # Utility functions
│   └── config.py            # Backend configuration
├── data/                    # Data storage (competitions, generated code)
├── frontend/
│   └── app.py               # Streamlit frontend application
├── log/                     # Application logs
├── output/                  # Generated submission files
├── .env.example             # Environment variable template
├── docker-compose.yml       # Docker composition
├── Dockerfile               # Docker build instructions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🚀 Quick Start

### 1) Prerequisites

* **Docker Desktop**: Ensure Docker is installed and running.
* **Kaggle API Key**: You need `kaggle.json` to get your username and key.
* **OpenAI API Key**: Required for LLM-based agents.

### 2) Configure Environment

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Fill in your `.env` file with your Kaggle credentials and OpenAI API key:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3) Build and Run with Docker Compose

```bash
# Build and start the services
docker-compose up --build
```

This command will:
1. Build the Docker image with all dependencies (Python 3.10, LightGBM, etc.).
2. Start the Streamlit frontend application.

### 4) Access the Application

Once the container is running, open your browser and navigate to:

`http://localhost:8501`

## 🧭 Usage

1. Open `http://localhost:8501`
2. Enter a Kaggle competition link or name (e.g. `https://www.kaggle.com/competitions/store-sales-time-series-forecasting`)
3. Select the AI architecture (e.g. **ReAct** or **RAG**)
4. Click **Fetch & Analyze Competition**
5. View the summary of the kaggle problem data
6. Click **Run Agent 🚀**
5. View real-time logs, generated code, and outputs
6. Download the resulting `submission.csv` or the generated code

## 📊 Evaluation Metrics
* Final prediction accuracy  (Kaggle leaderboard)
* Task success rate (Percentage of successful runs)
* Quality of data preprocessing (Evaluation of missing values, categorical variable handling)
* Feature engineering quality (evaluating new features)
* Execution time (overall and by stage)
* Explainability (feature importance, visualizations, text reports)
* Autonomy (human interaction count, tool usage, retry rate)
* Code complexity (lines, dependencies, cyclomatic complexity)
* Resource usage (memory, CPU/GPU time, LLM tokens)

## ❓ FAQ

**Q1: What manual setup is required?**

A: You must create a `.env` file and populate it with your personal credentials (Kaggle API keys, OpenAI API key). This step cannot be automated and must be completed before running the application.

**Q2: Why are Live Logs not showing for RAG and DLP Agents?**

A: These agents do not generate real-time execution logs in the same way as other agents. The process is running in the background and may take some time (approximately 1 minute) to generate results. Please be patient while the system processes your request.
