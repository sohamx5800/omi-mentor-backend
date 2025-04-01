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

# Initialize Summarization Model
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
    """Fetch response from Groq API with direct answers/suggestions."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    conn = http_client.HTTPSConnection(URL, context=context)
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Always provide a direct answer or suggestion without asking additional questions."},
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
        concise_response = summarize_text(full_response)  
        return concise_response
    else:
        return f"❌ Error {res.status}: {data.decode('utf-8')}"

def summarize_text(text):
    """Summarizes response while avoiding input length issues."""
    if len(text) <= 50:  # If text is already short, return as is
        return text  
    try:
        adjusted_max_length = max(len(text) // 2, 30)  # Ensure summarization doesn't fail
        summary = summarizer(text, max_length=adjusted_max_length, min_length=20, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"Summarization failed: {e}")
        return text  # Fallback to original text

def translate_to_english(text):
    """Automatically detects and translates text to English if necessary."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return original if translation fails

@app.post("/livetranscript")
async def live_transcription(request: Request):
    """Processes transcription, translates if necessary, and provides summarized response."""
    try:
        data = await request.json()
        segments = data.get("segments", [])
        if not segments:
            return {"message": "No transcription received"}

        transcript = " ".join(segment["text"] for segment in segments if "text" in segment).strip()
        if not transcript:
            return {"message": "No valid transcription received"}
        
        translated_text = translate_to_english(transcript)
        mood, suggestion = "Neutral 😐", "Stay focused and keep moving!"
        ai_response = ask_groq(translated_text)
        notification_message = summarize_text(ai_response)

        return {
            "message": notification_message,  
            "sentiment": mood,
            "response": ai_response
        }
    except Exception as e:
        print(f"Error in /livetranscript: {e}")
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
        mood, suggestion = "Neutral 😐", "Stay focused and keep moving!"
        ai_response = ask_groq(translated_text)

        return {"message": "Webhook received", "sentiment": mood, "response": ai_response}
    except Exception as e:
        print(f"Error in /webhook: {e}")
        return {"message": "Internal Server Error"}

# Task Management Routes
@app.get("/tasks")
def get_tasks(db: Session = Depends(SessionLocal)):
    tasks = db.query(Task).all()
    return {"tasks": tasks}

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
