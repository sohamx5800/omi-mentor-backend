from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import json
import ssl
import http.client as http_client
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline  
from deep_translator import GoogleTranslator
from sqlalchemy.orm import Session
from database import SessionLocal, init_db, Task
from config import API_KEY

# Download necessary NLTK data

sia = SentimentIntensityAnalyzer()

# Initialize Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
        concise_response = summarize_text(full_response, max_length=200)  
        return concise_response
    else:
        return f"❌ Error {res.status}: {data.decode('utf-8')}"

def summarize_text(text, max_length=100):
    """Summarizes response into 2-3 lines, keeping the core topic/suggestion."""
    if len(text) <= max_length:
        return text  
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]["summary_text"]
    return summary

def translate_to_english(text):
    """Automatically detects and translates text to English if necessary."""
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text
    except Exception:
        return text  # Return original text if translation fails

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
        sentiment_score = sia.polarity_scores(translated_text)["compound"]
        mood, suggestion = analyze_sentiment(sentiment_score)

        ai_response = ask_groq(translated_text)
        notification_message = summarize_text(ai_response, max_length=200)

        return {
            "message": notification_message,  
            "sentiment": mood,
            "response": ai_response
        }
    except Exception as e:
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
        sentiment_score = sia.polarity_scores(translated_text)["compound"]
        mood, suggestion = analyze_sentiment(sentiment_score)
        ai_response = ask_groq(translated_text)

        return {"message": "Webhook received", "sentiment": mood, "response": ai_response}
    except Exception as e:
        return {"message": "Internal Server Error"}

def analyze_sentiment(score):
    """Analyzes sentiment and provides a brief suggestion."""
    if score > 0.2:
        return "Happy 😊", "Keep up the great energy!"
    elif score < -0.2:
        return "Sad 😞", "Stay positive! You got this!"
    else:
        return "Neutral 😐", "Stay focused and keep moving!"

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
