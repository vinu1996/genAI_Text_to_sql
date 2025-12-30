🗣️ SQL-Speak: Natural Language to PostgreSQL Engine
SQL-Speak is an intelligent interface designed to democratize data access. It allows anyone—from marketing managers to product owners—to talk directly to a PostgreSQL database using everyday language.

By combining LangChain’s orchestration with Gemini’s reasoning and Ragas' evaluation, this project ensures that "Natural Language" doesn't just mean "guesses," but precise, executable code.


✨ The "Secret Sauce"
What makes this project different from a standard LLM prompt?

Postgres-Specific Intelligence: Automatically handles the "Case-Sensitivity Trap" by quoting identifiers like "TableName".

Dynamic Type Casting: Recognizes when orderdate is a string and injects ::timestamp or ::date logic on the fly.

Zero-Markdown Enforcement: Our custom prompt engineering ensures the LLM returns pure SQL, making it safe for programmatic execution.

Judge-Evaluator Loop: Uses a separate Groq-powered judge to audit the performance of the Gemini-powered generator.

🏗️ System Architecture
Context Injection: The system fetches the latest DDL from your database so the LLM always knows your current table structure.

The Generator: Uses Gemini-2.5-Flash-Lite for an optimal balance of speed and complex SQL reasoning.

The Auditor: A separate pipeline uses Gemma2 via Groq to run a "blind audit" on the generated queries.

🚦 Getting Started
1. Installation
Bash

pip install langchain-google-genai langchain-groq langchain-huggingface \
            langchain-community langchain-classic ragas psycopg2-binary torch
2. Configure Your Secrets
Create a .env file or export your keys:

Python

# Use environment variables for security!
GOOGLE_API_KEY = "YOUR_GEMINI_KEY"
GROQ_API_KEY = "YOUR_GROQ_KEY"
DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/aidb"
🧪 Quality Assurance (Evaluation)
We use RAGAS to ensure the system stays accurate. Every query is graded on:

Faithfulness: Does the SQL stay true to the schema?

Helpfulness: Based on a custom 5-point rubric, does the answer actually solve the user's intent?

Maliciousness: An automated "Safety Critic" that flags queries attempting to deceive or harm the data.

🚀 Roadmap & Future Works
The journey doesn't end at simple queries. Here is where we are going:

[ ] Self-Healing SQL: If a query fails, the agent should read the error message and try to fix the SQL automatically.

[ ] Vectorized Schema Mapping: For databases with 100+ tables, we will use a Vector DB to only "show" the LLM the most relevant tables for a given question.

[ ] Multi-Dialect Support: Expanding logic to support Snowflake, BigQuery, and Redshift specific quirks.

[ ] Explainer Mode: A toggle that explains how the query was built in plain English for educational purposes.
