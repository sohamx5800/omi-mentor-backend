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
        return f"Error {res.status}", "Retry"

def summarize_text(text, target_length=50):
    """Summarizes text to target_length characters with t5-small, ensuring coherence."""
    if len(text) <= target_length:
        return text
    try:
        # Use low max_length for concise output, adjust min_length for flexibility
        summary = summarizer(text, max_length=12, min_length=5, do_sample=False)[0]["summary_text"]
        # Truncate at last space to avoid mid-word cuts, add ellipsis
        if len(summary) > target_length:
            truncated = summary[:target_length-3].rsplit(" ", 1)[0] + "..."
            return truncated if len(truncated) <= target_length else summary[:target_length-3] + "..."
        return summary
    except Exception:
        # Fallback: truncate at last space with ellipsis
        truncated = text[:target_length-3].rsplit(" ", 1)[0] + "..."
        return truncated if len(truncated) <= target_length else text[:target_length-3] + "..."

def translate_to_english(text):
    """Automatically detects and translates text to English if necessary."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

@app.post("/livetranscript")
async def live_transcription(request: Request):
    """Processes transcription, provides answer and suggestion."""
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
        
        # Summarize answer and suggestion for notification (max 50 chars)
        short_answer = summarize_text(answer, 30)  # Room for suggestion
        short_suggestion = summarize_text(suggestion, 15)
        notification = f"{short_answer} | {short_suggestion}"[:50]

        return {
            "message": notification,  # Max 50 chars
            "response": answer,       # Full answer
            "suggestion": suggestion  # Full suggestion (1 word or up to 3 lines)
        }
    except Exception:
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