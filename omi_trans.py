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
import asyncio

# Initialize Summarization Model (Lightweight t5-small)
summarizer = pipeline("summarization", model="t5-small")

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

URL = "api.groq.com"
ENDPOINT = "/openai/v1/chat/completions"

def ask_groq(question):
    """Fetch response from Groq API with answer and suggestion."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    conn = http_client.HTTPSConnection(URL, context=context)
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Act as a mentor. Provide a complete, concise answer (1-2 sentences) followed by a suggestion (one word or up to three lines). Separate answer and suggestion with '||'."},
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
        full_response = json.loads(data.decode("utf-8"))["choices"][0]["message"]["content"]
        try:
            answer, suggestion = full_response.split("||", 1)
            return answer.strip(), suggestion.strip()
        except ValueError:
            return full_response.strip(), "Reflect"
    else:
        return "Error fetching response", "Retry"

def summarize_text(text, target_length=50):
    """Summarizes text quickly to target_length characters with t5-small."""
    if len(text) <= target_length:
        return text
    # Fallback to simple truncation for speed if summarization is slow
    truncated = text[:target_length-3].rsplit(" ", 1)[0] + "..."
    try:
        # Use very low max_length for fast, concise output
        summary = summarizer(text, max_length=12, min_length=5, do_sample=False)[0]["summary_text"]
        if len(summary) > target_length:
            return summary[:target_length-3].rsplit(" ", 1)[0] + "..."
        return summary
    except Exception:
        return truncated if len(truncated) <= target_length else text[:target_length-3] + "..."

def translate_to_english(text):
    """Automatically detects and translates text to English if necessary."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

@app.post("/livetranscript")
async def live_transcription(request: Request):
    """Processes transcription, provides answer and suggestion with timeout."""
    try:
        data = await request.json()
        segments = data.get("segments", [])
        if not segments:
            return {"message": "No transcription received"}

        transcript = " ".join(segment["text"] for segment in segments if "text" in segment).strip()
        if not transcript:
            return {"message": "No valid transcription received"}
        
        translated_text = translate_to_english(transcript)
        answer, suggestion = ask_groq(translated_text)
        
        # Summarize with a timeout to prevent 499
        async def summarize_with_timeout(text, length):
            try:
                return await asyncio.wait_for(asyncio.to_thread(summarize_text, text, length), timeout=2.0)
            except asyncio.TimeoutError:
                return text[:length-3].rsplit(" ", 1)[0] + "..."

        short_answer = await summarize_with_timeout(answer, 30)
        short_suggestion = await summarize_with_timeout(suggestion, 15)
        notification = f"{short_answer} | {short_suggestion}"[:50]

        return {
            "message": notification,  # Max 50 chars
            "response": answer,       # Full answer
            "suggestion": suggestion  # Full suggestion
        }
    except Exception as e:
        return {"message": "Server error", "error": str(e)}

@app.post("/webhook")
async def receive_transcription(request: Request):
    """Webhook endpoint to receive transcriptions."""
    try:
        data = await request.json()
        transcript = data.get("transcript", "").strip()
        if not transcript:
            return {"message": "No transcription received"}

        translated_text = translate_to_english(transcript)
        answer, suggestion = ask_groq(translated_text)
        return {
            "message": "Webhook received",
            "response": answer,
            "suggestion": suggestion
        }
    except Exception:
        return {"message": "Internal Server Error"}

# Task Management Routes
@app.get("/tasks")
def get_tasks(db: Session = Depends(SessionLocal)):
    return {"tasks": db.query(Task).all()}

@app.post("/tasks")
def add_task(task_text: str, db: Session = Depends(SessionLocal)):
    new_task = Task(task=task_text)
    db.add(new_task)
    db.commit()
    return {"message": "Task added successfully!"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(SessionLocal)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        db.delete(task)
        db.commit()
        return {"message": "Task deleted successfully!"}
    return {"error": "Task not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)