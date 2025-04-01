from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import json
import ssl
import http.client as http_client
from transformers import pipeline
from deep_translator import GoogleTranslator
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, Task
from config import API_KEY
import logging

# Initialize Database
init_db()

# FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API setup
URL = "api.groq.com"
ENDPOINT = "/openai/v1/chat/completions"

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load T5 Pipeline for Summarization & Refinement
t5_pipeline = pipeline("summarization", model="t5-small")

def ask_groq(question):
    """Fetch response from Groq API."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    conn = http_client.HTTPSConnection(URL, context=context)
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Act as a mentor. Provide clear answers and suggestions."},
            {"role": "user", "content": question}
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    conn.request("POST", ENDPOINT, body=json.dumps(payload), headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    if res.status == 200:
        return json.loads(data.decode("utf-8"))["choices"][0]["message"]["content"]
    else:
        return f"❌ Error {res.status}: {data.decode('utf-8')}"

def refine_text_with_t5(text, max_length=50):
    """Uses T5 to summarize and refine responses."""
    if len(text) <= max_length:
        return text  # If short, return as is
    
    try:
        summary = t5_pipeline(text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return text  # Fallback to original text

def translate_to_english(text):
    """Translates text to English if necessary."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text  # Fallback to original text

@app.post("/livetranscript")
async def live_transcription(request: Request):
    """Processes transcription, translates if needed, and sends refined response."""
    try:
        data = await request.json()
        segments = data.get("segments", [])
        if not segments:
            return {"message": "No transcription received"}

        transcript = " ".join(segment["text"] for segment in segments if "text" in segment).strip()
        if not transcript:
            return {"message": "No valid transcription received"}
        
        translated_text = translate_to_english(transcript)
        ai_response = ask_groq(translated_text)
        refined_response = refine_text_with_t5(ai_response)

        return {
            "message": refined_response,
            "response": ai_response,
            "suggestion": "✅ Check full response in the app." if len(ai_response) > 50 else ai_response
        }
    except Exception as e:
        logging.error(f"Error in live_transcription: {e}")
        return {"message": "Internal Server Error"}

@app.post("/webhook")
async def receive_transcription(request: Request):
    """Webhook endpoint to receive transcriptions."""
    try:
        data = await request.json()
        transcript = data.get("transcript", "").strip()
        if not transcript:
            return {"message": "No transcription received"}

        translated_text = translate_to_english(transcript)
        ai_response = ask_groq(translated_text)
        refined_response = refine_text_with_t5(ai_response)

        return {
            "message": "Webhook received",
            "response": ai_response,
            "suggestion": "✅ View full response in the app." if len(ai_response) > 50 else ai_response
        }
    except Exception as e:
        logging.error(f"Error in webhook: {e}")
        return {"message": "Internal Server Error"}

# Task Management API
@app.get("/tasks")
def get_tasks(db: Session = Depends(SessionLocal)):
    """Retrieve all tasks from the database."""
    try:
        tasks = db.query(Task).all()
        return {"tasks": tasks}
    except Exception as e:
        logging.error(f"Error in get_tasks: {e}")
        return {"message": "Error retrieving tasks"}

@app.post("/tasks")
def add_task(task_text: str, db: Session = Depends(SessionLocal)):
    """Add a new task to the database."""
    try:
        new_task = Task(task=task_text)
        db.add(new_task)
        db.commit()
        return {"message": "Task added successfully!"}
    except Exception as e:
        logging.error(f"Error in add_task: {e}")
        return {"message": "Error adding task"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(SessionLocal)):
    """Delete a task from the database by task_id."""
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            db.delete(task)
            db.commit()
            return {"message": "Task deleted successfully!"}
        return {"error": "Task not found"}
    except Exception as e:
        logging.error(f"Error in delete_task: {e}")
        return {"message": "Error deleting task"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
