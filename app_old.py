from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests, os, uuid, threading, sqlite3, numpy as np, faiss, json, html, time, datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ===================================
# 🔍 Debug logging
# ===================================
LOG_FILE = os.path.join(os.getcwd(), "psyduck_intent_log.txt")

def log_event(event_type, message):
    """Append structured log events to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{event_type.upper()}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# ===================================
# 🔧 CONFIGURATION
# ===================================
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "0.0.0.0")  # Windows IP for WSL
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

MAX_TURNS = 12
MAX_MESSAGE_LENGTH = 1500

DB_PATH = os.path.join(os.getcwd(), "data", "fashion.db")
INDEX_PATH = "data/fashion.index"
EMBED_MODEL = "all-mpnet-base-v2"

# ===================================
# 🧠 LOAD COMPONENTS
# ===================================
print("🧠 Loading outfit recommender...")
# Connect to DB in a way that's safe for threading if check_same_thread=False
# For this app, we'll connect in each function, which is safer.
model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
print("✅ Recommender ready!")

# ===================================
# 🚀 FASTAPI SETUP
# ===================================
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

image_dir = os.path.join(os.getcwd(), "data/database")
if os.path.exists(image_dir):
    app.mount("/images", StaticFiles(directory=image_dir), name="images")
else:
    print("⚠️ Warning: 'data/database' folder not found. Images won't be served.")

conversation_store = {}
store_lock = threading.Lock()

# ===================================
# 🧩 HELPERS
# ===================================
def trim_text(text: str) -> str:
    return (text[:MAX_MESSAGE_LENGTH] + "...") if len(text) > MAX_MESSAGE_LENGTH else text

def append_message(conv_id: str, role: str, text: str):
    text = trim_text(text)
    with store_lock:
        conversation_store.setdefault(conv_id, []).append({"role": role, "text": text})
        if len(conversation_store[conv_id]) > MAX_TURNS:
            conversation_store[conv_id] = conversation_store[conv_id][-MAX_TURNS:]

def build_prompt(conv_id: str, user_text: str) -> str:
    # This build_prompt is now only for "general chat"
    base = [
        "You are Psyduck — a witty but helpful fashion assistant.",
        "You answer general questions. If the user asks for an outfit, just chat naturally."
    ]
    with store_lock:
        history = conversation_store.get(conv_id, [])
    for m in history:
        role = "User" if m["role"] == "user" else "Psyduck"
        base.append(f"{role}: {m['text']}")
    base.append(f"User: {user_text}")
    base.append("Psyduck:")
    return "\n".join(base)

def parse_ollama_response(resp_text: str) -> str:
    """Safely parse Ollama NDJSON/stream output into a single text response."""
    try:
        data = json.loads(resp_text)
        if isinstance(data, dict):
            return data.get("response") or data.get("text") or ""
    except Exception:
        pass

    final = ""
    for line in resp_text.splitlines():
        try:
            obj = json.loads(line)
            final += obj.get("response", "")
        except Exception:
            continue
    return final.strip()

# ===================================
# 🧩 Intent Manager (multi-turn)
# ===================================
class IntentManager:
    REQUIRED_FIELDS = ["occasion", "formality", "gender", "season", "style"]

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def start(self, user_query: str):
        sid = str(uuid.uuid4())
        session = {
            "query": user_query,
            "intent": {f: None for f in self.REQUIRED_FIELDS},
            "current_field": self.REQUIRED_FIELDS[0],
            "created": time.time()
        }
        with self.lock:
            self.sessions[sid] = session
        return sid, session

    def get(self, session_id: str):
        with self.lock:
            return self.sessions.get(session_id)

    def set_field(self, session_id: str, field: str, value: str):
        with self.lock:
            sess = self.sessions.get(session_id)
            if not sess:
                return None
            sess["intent"][field] = value
            next_field = next((f for f in self.REQUIRED_FIELDS if not sess["intent"].get(f)), None)
            sess["current_field"] = next_field
            return sess

intent_manager = IntentManager()


# ===================================
# 🧭 FIELD DEFINITIONS & NORMALIZER
# ===================================
FIELD_DEFINITIONS = {
    "occasion": "the event type such as wedding, party, interview, picnic, or office.",
    "formality": "the dress code level such as casual, semi-formal, formal, or business.",
    "gender": "the intended wearer’s gender such as male, female, or unisex.",
    "season": "the weather or time of year such as summer, winter, spring, or autumn.",
    "style": "the overall fashion vibe such as sporty, elegant, streetwear, vintage, boho, or classic."
}

def normalize_field_value(field: str, val: str) -> str:
    """Normalize common synonyms for better FAISS filtering."""
    val = val.lower().strip()

    replacements = {
        "occasion": {
            "party outfit": "party",
            "birthday outfit": "birthday",
            "work": "office",
            "meeting": "office",
            "gym wear": "gym",
            "date night": "date",
            "friend's house": "social"
        },
        "formality": {
            "informal": "casual",
            "semi formal": "semi-formal",
            "business wear": "business",
            "smart": "semi-formal",
            "relaxed": "casual"
        },
        "gender": {
            "girl": "female",
            "woman": "female",
            "lady": "female",
            "boy": "male",
            "man": "male",
            "gentleman": "male",
            "anyone": "unisex",
            "everyone": "unisex"
        },
        "season": {
            "rainy": "monsoon",
            "cold": "winter",
            "hot": "summer"
        },
        "style": {
            "modern": "modern casual",
            "classy": "elegant",
            "athletic": "sporty",
            "retro": "vintage"
        }
    }
    
    # Handle direct field value replacement first
    if field in replacements:
        if val in replacements[field]:
            return replacements[field][val]
            
    # Handle partial string matching
    for key, mapping in replacements.items():
        if key == field:
            for k, v in mapping.items():
                if k in val:
                    return v
    return val

# ===================================
# 🧠 INTENT HELPER (Reusable Logic v2)
# ===================================
# ===================================
# 🧠 INTENT HELPER (Reusable Logic v3 - Few-Shot)
# ===================================
async def begin_intent_flow(user_query: str):
    """
    Reusable helper function to start an intent session and pre-fill.
    This version uses a "few-shot" prompt for better extraction.
    """
    if not user_query:
        return {"error": "Empty query."}

    # 🧠 Step 1: Create a new session
    session_id, session = intent_manager.start(user_query)
    log_event("session_start", f"Session={session_id} | Query='{user_query}'")

    # 🧠 Step 2: Try to "pre-fill" the intent (with a better prompt)
    extract_prompt = (
        f"You are an expert fashion entity extractor. Your job is to fill a JSON object based on the user's request. Leave any unmentioned fields as null.\n"
        f"--- FIELDS ---\n"
        " occasion: (e.g., 'wedding', 'party', 'office')\n"
        " formality: (e.g., 'casual', 'formal', 'semi-formal')\n"
        " gender: (e.g., 'male', 'female', 'unisex')\n"
        " season: (e.g., 'summer', 'winter', 'rainy')\n"
        " style: (e.g., 'sporty', 'elegant', 'boho', 'vintage')\n"
        f"--- END FIELDS ---\n\n"
        
        f"--- Example 1 ---\n"
        "Request: \"i need a suit for a winter wedding\"\n"
        "JSON: {{\"occasion\": \"wedding\", \"formality\": \"formal\", \"gender\": null, \"season\": \"winter\", \"style\": \"formal\"}}\n\n"
        
        f"--- Example 2 ---\n"
        "Request: \"trim look, gender-neutral, rainy, no rules, red\"\n"
        "JSON: {{\"occasion\": null, \"formality\": \"casual\", \"gender\": \"unisex\", \"season\": \"rainy\", \"style\": \"trim look\"}}\n\n"

        f"--- Example 3 ---\n"
        "Request: \"something for the gym\"\n"
        "JSON: {{\"occasion\": \"gym\", \"formality\": \"casual\", \"gender\": null, \"season\": null, \"style\": \"sporty\"}}\n\n"
        
        f"--- CURRENT TASK ---\n"
        f"Request: \"{user_query}\"\n"
        "JSON:"
    )
    
    raw_res = "{}" # Default
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": extract_prompt, "stream": False, "format": "json"},
            timeout=60
        )
        r.raise_for_status()
        
        # Handle cases where the model might return a non-JSON string
        raw_res = r.json().get("response", "{}")
        if not raw_res.startswith("{"):
            raw_res = "{" + raw_res.split("{", 1)[-1]
        if not raw_res.endswith("}"):
            raw_res = raw_res.rsplit("}", 1)[0] + "}"

        extracted_data = json.loads(raw_res.strip())

        # 🧠 Step 3: Fill the session with extracted values
        filled_count = 0
        for field in IntentManager.REQUIRED_FIELDS:
            val = extracted_data.get(field)
            if val and str(val).lower() not in ["null", "none", "unknown"]:
                norm_val = normalize_field_value(field, str(val))
                intent_manager.set_field(session_id, field, norm_val)
                filled_count += 1
        
        session = intent_manager.get(session_id) # Get updated session
        log_event("prefill", f"Session={session_id} | Filled {filled_count} fields | Intent={session['intent']}")

    except Exception as e:
        log_event("prefill_error", f"Session={session_id} | Error={e} | Response='{raw_res}'")
        pass # Continue anyway

    # 🧠 Step 4: Find the first *missing* field
    current_field = session["current_field"]

    if not current_field:
        # Everything was pre-filled!
        log_event("intent_complete", f"Session={session_id} | Completed via pre-fill | Final={session['intent']}")
        return {"complete": True, "intent": session["intent"], "session_id": session_id, "question": "Thanks! I've got everything I need!"}

    # 🧠 Step 5: Ask the first question for the *missing* field (with new, direct prompt)
    ask_prompt = (
        f"You are Psyduck, a helpful assistant.\n"
        f"You *must* ask the user for *only* this one field: '{current_field}'.\n"
        f"This field means: {FIELD_DEFINITIONS.get(current_field, 'a fashion detail')}.\n"
        f"Ask a short, direct question to get this information. Do not add any other topics.\n"
        f"Example: 'What {current_field} are you looking for?'\n"
        f"Your question:"
    )
    
    q = f"What {current_field} are you looking for?" # Default
    try:
        r_q = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": ask_prompt, "stream": False},
            timeout=60
        )
        r_q.raise_for_status()
        q = (r_q.json().get("response") or q).strip().splitlines()[0].strip()

    except Exception as e:
        log_event("intent_start_error", f"Session={session_id} | Field={current_field} | Error={e}")
        pass

    log_event("intent_first_question", f"Session={session_id} | Field={current_field} | Question='{q}'")

    return {
        "session_id": session_id,
        "question": q,
        "intent": session["intent"],
        "complete": False
    }
# ===================================
# 🌐 ROUTES
# ===================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---- Chat (with Triage) ----
class ChatInput(BaseModel):
    text: str
    conversation_id: str | None = None

# ---- Chat (with Triage v2) ----
class ChatInput(BaseModel):
    text: str
    conversation_id: str | None = None
# ---- Chat (with Triage v2) ----
class ChatInput(BaseModel):
    text: str
    conversation_id: str | None = None

@app.post("/chat")
async def chat(data: ChatInput):
    conv_id = data.conversation_id or str(uuid.uuid4())
    user_text = data.text.strip()
    if not user_text:
        return {"reply": "Please say something 🐤", "conversation_id": conv_id}

    # Get conversation history for context
    with store_lock:
        history = conversation_store.get(conv_id, [])
    
    # Get the last 2 turns as context
    history_context = []
    if len(history) > 0:
        for m in history[-2:]: # Get last 2 messages
            role = "User" if m["role"] == "user" else "Bot"
            history_context.append(f"{role}: {m['text']}")
    
    context_str = "\n".join(history_context)
    append_message(conv_id, "user", user_text) # Append the new message *after* getting history

    # 🧠 Step 1: Triage (Smarter, with context)
    triage_prompt = (
        f"You are a conversation router. Here is the end of a conversation:\n"
        f"--- CONTEXT ---\n{context_str}\n--- END CONTEXT ---\n\n"
        f"New User Message: \"{user_text}\"\n\n"
        "Is this New User Message starting a *brand new* outfit request, or is it a follow-up chat, a question about previous results, or a general comment?\n"
        "Respond with only 'NEW_REQUEST' or 'FOLLOW_UP'."
    )
    
    is_new_request = False
    try:
        r_triage = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": triage_prompt, "stream": False},
            timeout=60
        )
        r_triage.raise_for_status()
        res_text = (r_triage.json().get("response") or "FOLLOW_UP").strip().upper()
        
        if res_text.startswith("NEW_REQUEST"):
            is_new_request = True
        else:
            is_new_request = False # Default to follow-up
            
        log_event("chat_triage", f"Conv={conv_id} | Text='{user_text}' | Triage='{res_text}'")
    except Exception as e:
        log_event("chat_triage_error", f"Conv={conv_id} | Error={e}")
        is_new_request = False # Fail safe

    # 🧠 Step 2: Handle based on Triage
    
    # --- PATH A: It's a NEW outfit request -> Start intent flow ---
    if is_new_request:
        log_event("chat_to_intent", f"Conv={conv_id} | Starting new intent flow for: '{user_text}'")
        
        intent_data = await begin_intent_flow(user_query=user_text)
        
        if "error" in intent_data:
             return {"reply": f"⚠️ Error starting intent flow: {intent_data['error']}", "conversation_id": conv_id}
        
        reply = intent_data.get("question", "So, what's the occasion?")
        append_message(conv_id, "bot", reply)
        
        return {
            "reply": reply,
            "conversation_id": conv_id,
            "intent_session_id": intent_data.get("session_id"),
            "intent": intent_data.get("intent"),
            "intent_complete": intent_data.get("complete", False)
        }

    # --- PATH B: It's just a FOLLOW_UP or general chat ---
    prompt = build_prompt(conv_id, user_text) # build_prompt now includes the new user message
    try:
        res = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=60)
        
        # --- THIS IS THE FIX ---
        res.raise_for_status() # Was 'res_raise_for_status()'
        # --- END OF FIX ---
        
        reply = parse_ollama_response(res.text) or "Hmm… Psyduck’s having a tiny brain fog."
        append_message(conv_id, "bot", reply)
        
        return {"reply": reply, "conversation_id": conv_id}
        
    except Exception as e:
        return {"reply": f"⚠️ Error: {e}", "conversation_id": conv_id}
# ===================================
# 🧠 INTENT START
# ===================================
# ---- Intent ----
@app.post("/intent/start")
async def intent_start(body: dict):
    """
    Starts a multi-turn intent session. (Now just a wrapper)
    """
    user_query = (body.get("query") or "").strip()
    result = await begin_intent_flow(user_query)
    
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)
    
    return result

# ===================================
# 🧠 INTENT ANSWER (with Triage v3 - Few-Shot)
# ===================================
@app.post("/intent/answer")
async def intent_answer(body: dict):
    """
    Accept the user's answer for the current field of a session.
    This version includes robust "few-shot" triage logic.
    """
    session_id = body.get("session_id")
    answer = (body.get("answer") or "").strip()

    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    session = intent_manager.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    current_field = session["current_field"]
    if not current_field:
        return {"complete": True, "intent": session["intent"]}

    # 🧠 Step 1: Triage - Is this an answer or a question?
    # This is the NEW, robust "few-shot" prompt for /intent/answer
    triage_prompt = (
        f"You are a text classifier. Your *only* job is to output one of two words: 'QUESTION' or 'ANSWER'.\n\n"
        
        "A 'QUESTION' is when the user is confused, asks for a definition, or asks *about* the field (e.g., 'what does that mean?', 'why?', 'what is formality?').\n"
        "An 'ANSWER' is when the user provides a value *for* the field (e.g., 'male', 'casual', 'party', 'summer').\n\n"
        
        "--- Examples ---\n"
        "Bot: What formality?\n"
        "User: what's that?\n"
        "Your classification: QUESTION\n\n"
        
        "Bot: What gender?\n"
        "User: male\n"
        "Your classification: ANSWER\n\n"
        
        "Bot: What season?\n"
        "User: i'm not sure\n"
        "Your classification: ANSWER\n\n"
        
        "Bot: What style?\n"
        "User: casual\n"
        "Your classification: ANSWER\n"
        "--- End Examples ---\n\n"
        
        "--- CURRENT TASK ---\n"
        f"Bot: What {current_field}?\n"
        f"User: \"{answer}\"\n"
        "Your classification:"
    )
    
    triage_result = "ANSWER" # Default
    try:
        r_triage = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": triage_prompt, "stream": False},
            timeout=60
        )
        r_triage.raise_for_status()
        
        # Use stricter parsing
        res_text = (r_triage.json().get("response") or "ANSWER").strip().upper()
        if res_text.startswith("QUESTION"):
            triage_result = "QUESTION"
        else:
            triage_result = "ANSWER"
            
        log_event("triage", f"Session={session_id} | Field={current_field} | Answer='{answer}' | Triage='{triage_result}'")

    except Exception as e:
        log_event("triage_error", f"Session={session_id} | Field={current_field} | Error={e}")
        triage_result = "ANSWER" # Fail-safe: assume it's an answer

    # 🧠 Step 2: Handle based on Triage
    
    # --- PATH A: User is asking a QUESTION ---
    if triage_result == "QUESTION":
        log_event("clarification", f"Session={session_id} | Field='{current_field}' | User is confused: '{answer}'")

        clarify_prompt = (
            f"You are Psyduck, a helpful fashion assistant.\n"
            f"You are trying to get the field '{current_field}'.\n"
            f"The user was confused and asked: \"{answer}\".\n"
            f"First, politely answer their question using this definition: '{FIELD_DEFINITIONS.get(current_field, 'a detail about your outfit')}'\n"
            f"Second, *re-ask* the original question for '{current_field}' in a clear, direct way.\n"
            "Only output your helpful response."
        )

        try:
            r2 = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": clarify_prompt, "stream": False},
                timeout=60
            )
            r2.raise_for_status()
            next_q = (r2.json().get("response") or "My mistake!").strip()
            
        except Exception as e:
            log_event("clarify_error", f"Session={session_id} | Field={current_field} | Error={e}")
            next_q = f"Sorry, I wasn't clear. {FIELD_DEFINITIONS.get(current_field)}. So, what is the {current_field}?"

        return {
            "complete": False,
            "next_field": current_field, # Stay on the *same* field
            "question": next_q,
            "intent": session["intent"],
            "note": "clarification_provided"
        }

    # --- PATH B: User gave an ANSWER ---
    # (If triage did not return, we are on the ANSWER path)
    
    interpret_prompt = (
        f"You are a short-answer entity extractor.\n"
        f"The bot asked the user for the field: '{current_field}' (which means: {FIELD_DEFINITIONS.get(current_field, 'a user detail')}).\n"
        f"The user answered: \"{answer}\"\n\n"
        "What is the single best value for the '{current_field}' field based *only* on the user's answer?\n"
        "Respond with only that value (e.g., 'birthday', 'formal', 'male').\n"
        "If the answer is irrelevant or unclear, respond with the single word 'UNKNOWN'."
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": interpret_prompt, "stream": False},
            timeout=60
        )
        r.raise_for_status()
        val = (r.json().get("response") or "UNKNOWN").strip().strip('"\'.,!?').lower()

        invalids = ["unknown", "i don't know", "not sure", "none", "no idea", "maybe", "idk", "", "yes", "ok", "okay", "sure"]
        if val in invalids or len(val.split()) > 5:
            val = "UNKNOWN"

        log_event("extract", f"Session={session_id} | Field={current_field} | Answer='{answer}' | Extracted='{val}'")

    except Exception as e:
        log_event("extract_error", f"Session={session_id} | Field={current_field} | Error={e}")
        val = "UNKNOWN"

    # If extraction fails, re-ask (this is different from user confusion)
    if val.upper() == "UNKNOWN":
        log_event("extract_failed", f"Session={session_id} | Field='{current_field}' | Answer='{answer}'")
        
        next_q = f"Hmm, I didn't quite catch that for {current_field}. Could you try again? (This refers to: {FIELD_DEFINITIONS.get(current_field)})"
        
        return {
            "complete": False,
            "next_field": current_field, # Stay on the *same* field
            "question": next_q,
            "intent": session["intent"],
            "note": "extraction_failed"
        }

    # 🧠 Step 3: Normalize and set the field
    val = normalize_field_value(current_field, val)
    intent_manager.set_field(session_id, current_field, val)
    session = intent_manager.get(session_id)
    next_field = session["current_field"]

    if not next_field:
        # Completed all fields
        final_intent = session["intent"]
        log_event("intent_complete", f"Session={session_id} | Final={final_intent}")
        return {"complete": True, "intent": final_intent, "session_id": session_id}

    # 🧠 Step 4: Ask model to generate next field question (with new, direct prompt)
    ask_prompt = (
        f"You are Psyduck, a helpful assistant.\n"
        f"You *must* ask the user for *only* this one field: '{next_field}'.\n"
        f"This field means: {FIELD_DEFINITIONS.get(next_field, 'a fashion detail')}.\n"
        f"Ask a short, direct question to get this information. Do not add any other topics.\n"
        f"Example: 'What {next_field} are you looking for?'\n"
        f"Your question:"
    )

    try:
        r3 = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": ask_prompt, "stream": False},
            timeout=60
        )
        r3.raise_for_status()
        next_q = (r3.json().get("response") or f"Could you provide the {next_field}?").strip().splitlines()[0].strip()

    except Exception as e:
        log_event("next_field_error", f"Session={session_id} | Field={next_field} | Error={e}")
        next_q = f"What about the {next_field}?"

    return {
        "complete": False,
        "next_field": next_field,
        "question": next_q,
        "intent": session["intent"]
    }

# ===================================
# 🧠 RECOMMEND (High-Performance)
# ===================================
@app.post("/recommend", response_class=JSONResponse)
async def recommend(data: dict):
    query = (data.get("query") or "").strip()
    intent = data.get("intent") or {}
    if not query:
        return JSONResponse({"error": "Empty query."}, status_code=400)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Step 1: Filter (SQL) ---
    # Get the *candidate IDs* that match the filters (e.g., "formal", "party")
    filters, params = [], []
    for k, v in intent.items():
        if v:
            filters.append("(description LIKE ? OR category LIKE ?)")
            params += [f"%{v}%", f"%{v}%"]

    where_clause = " OR ".join(filters) if filters else "1=1"
    
    # We only need the IDs from SQL
    sql = f"SELECT id FROM products WHERE {where_clause} LIMIT 300;"
    try:
        cursor.execute(sql, params)
    except sqlite3.OperationalError as e:
        log_event("sql_error", f"Error executing filter SQL: {e} | SQL: {sql} | PARAMS: {params}")
        conn.close()
        return {"query": query, "intent": intent, "count": 0, "results": []}
        
    candidate_ids = [r[0] for r in cursor.fetchall()]

    if not candidate_ids:
        conn.close()
        return {"query": query, "intent": intent, "count": 0, "results": []}

    # --- Step 2: Rerank (FAISS) ---
    
    # Encode the user's query *once*
    qv = model.encode([query]).astype("float32")

    sel = faiss.IDSelectorBatch(np.array(candidate_ids, dtype='int64'))

    results = []
    try:
        # Search the *GLOBAL* index, passing the selector in 'params'
        k = min(10, len(candidate_ids))
        D, I = index.search(qv, k=k, params=faiss.SearchParameters(sel=sel))

        top_ids = [int(id) for id in I[0] if id != -1] # Get valid IDs
        
        if not top_ids:
             conn.close()
             return {"query": query, "intent": intent, "count": 0, "results": []}

        # --- Step 3: Fetch final results ---
        placeholders = ",".join(["?"] * len(top_ids))
        sql_final = f"SELECT id, image, `display name`, category, description FROM products WHERE id IN ({placeholders});"
        
        cursor.execute(sql_final, top_ids)
        rows_final = cursor.fetchall()

        # Re-order the results to match FAISS ranking
        id_to_row = {row[0]: row for row in rows_final}
        ordered_rows = [id_to_row[id] for id in top_ids if id in id_to_row]
        
        # Format the final output
        for id, image, name, cat, desc in ordered_rows:
            results.append({
                "id": id, 
                "image": f"/images/{os.path.basename(image)}", 
                "display_name": name, 
                "category": cat, 
                "description": desc
            })

    except Exception as e:
        conn.close()
        log_event("recommend_error", f"FAISS search with IDSelector failed. Is 'fashion.index' an IndexIDMap? Error: {e}")
        return JSONResponse({"error": "Sorry, an error occurred while searching."}, status_code=500)
    
    finally:
        conn.close()
        
    return {"query": query, "intent": intent, "count": len(results), "results": results}
#=============================
#       RESET CHAT
#=============================

class ResetInput(BaseModel):
    conversation_id: str | None = None

@app.post("/reset")
async def reset(data: ResetInput):
    """
    Clears a conversation history from the server's store.
    """
    if data.conversation_id:
        with store_lock:
            if data.conversation_id in conversation_store:
                del conversation_store[data.conversation_id]
                log_event("reset", f"Conv={data.conversation_id} cleared.")
                return {"status": "reset_success"}
    return {"status": "not_found_or_no_id"}


# ---- Logs ----
@app.get("/logs", response_class=PlainTextResponse)
async def view_logs():
    if not os.path.exists(LOG_FILE):
        return PlainTextResponse("🪶 No logs yet!", status_code=200)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return PlainTextResponse(content[-8000:] or "🪶 Log empty.", status_code=200)
    except Exception as e:
        return PlainTextResponse(f"⚠️ Error reading logs: {e}", status_code=500)

# ---- Health ----
@app.get("/health")
async def health():
    return {"status": "ok"}

# ===================================
# 🚀 RUN
# ===================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Psyduck web app running at http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)