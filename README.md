# LLM Brand Service (FastAPI + LangChain, JSON Storage)
This is a simplified FastAPI service for managing brand writing requirements via LLMs.
It uses JSON files instead of a database.

## Quickstart
1. Create and activate a virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run the API:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

4. Open http://localhost:8000/docs to test the endpoints.

## Endpoints
- `POST /upload-brand`: Upload brand guidebook (PDF or text) and extract requirements.
- `GET /brands`: List saved brands.
- `POST /chat`: Ask the LLM to proofread or write text with combined system + brand prompts.
