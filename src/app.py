from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests, os, sys, uuid, threading, sqlite3, numpy as np, faiss, json, html, time, datetime, re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# PATH & SYSTEM CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# Debug logging
LOG_FILE = os.path.join(PROJECT_DIR, "psyduck_intent_log.txt")

def log_event(event_type, message):
    """Append structured log events to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{event_type.upper()}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


# CONFIGURATION
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")  # Host IP for Ollama (127.0.0.1 or Windows host in WSL)
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

def get_ollama_model() -> str:
    """Return configured Ollama model from environment or fallback default."""
    return os.getenv("OLLAMA_MODEL") or OLLAMA_MODEL

MAX_TURNS = 12
MAX_MESSAGE_LENGTH = 1500

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "fashion.db")
INDEX_PATH = os.path.join(DATA_DIR, "fashion.index")
EMBED_MODEL = "all-mpnet-base-v2"


# LOAD COMPONENTS
print("[INFO] Loading outfit recommender...")
model = SentenceTransformer(EMBED_MODEL)
index = None
if os.path.exists(INDEX_PATH):
    try:
        index = faiss.read_index(INDEX_PATH)
        print("[INFO] Recommender index ready.")
    except Exception as e:
        print(f"[WARNING] Failed to load FAISS index from '{INDEX_PATH}': {e}")
else:
    print(f"[WARNING] 'fashion.index' not found at '{INDEX_PATH}'. Run data/build_database.py or run.sh.")


# FASTAPI SETUP
app = FastAPI(title="Psyduck Outfit Assistant")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

image_dir = os.path.join(DATA_DIR, "database")
if os.path.exists(image_dir):
    app.mount("/images", StaticFiles(directory=image_dir), name="images")
else:
    print(f"[WARNING] '{image_dir}' folder not found. Images will not be served.")

conversation_store = {}
store_lock = threading.Lock()


# HELPERS
def trim_text(text: str) -> str:
    return (text[:MAX_MESSAGE_LENGTH] + "...") if len(text) > MAX_MESSAGE_LENGTH else text

def append_message(conv_id: str, role: str, text: str):
    text = trim_text(text)
    with store_lock:
        conversation_store.setdefault(conv_id, []).append({"role": role, "text": text})
        if len(conversation_store[conv_id]) > MAX_TURNS:
            conversation_store[conv_id] = conversation_store[conv_id][-MAX_TURNS:]

def build_prompt(conv_id: str, user_text: str) -> str:
    """Build a structured conversational prompt for general chat/follow-ups."""
    base = [
        "System: You are Psyduck, a helpful, polite, and concise fashion styling assistant.",
        "System: Provide direct styling advice in 1 to 3 short sentences. Focus entirely on fashion and outfits.",
        "System: Do not invent fictional user turns or repeat system prompts."
    ]
    with store_lock:
        history = conversation_store.get(conv_id, [])
    for m in history[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        base.append(f"{role}: {m['text']}")
    base.append(f"User: {user_text}")
    base.append("Assistant:")
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


# Intent Manager (multi-turn)
class IntentManager:
    # UPDATED: Removed 'season', added 'category' and 'color'
    REQUIRED_FIELDS = ["occasion", "category", "formality", "gender", "color", "style"]

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



# FIELD DEFINITIONS & NORMALIZER

FIELD_DEFINITIONS = {
    "occasion": "the event type such as wedding, party, interview, picnic, date, gym, or office.",
    "category": "the clothing item type such as t-shirt, shirt, dress, suit, jeans, trousers, jacket, or shoes.",
    "formality": "the dress code level such as casual, semi-formal, formal, or business.",
    "gender": "the intended wearer's gender: male, female, or unisex.",
    "color": "the primary color of the item, such as black, white, blue, red, green, or navy.",
    "style": "the overall fashion aesthetic such as sporty, elegant, streetwear, vintage, boho, or classic."
}

FIELD_HINTS = {
    "occasion": "Valid examples include: wedding, party, birthday, office, date, gym, casual, vacation, interview, social.",
    "category": "Valid examples include: t-shirt, shirt, suit, dress, jeans, trousers, jacket, shoes, sneakers, skirt.",
    "formality": "Valid values are strictly: casual, semi-formal, formal, business.",
    "gender": "Valid values are strictly: male, female, unisex (or man, woman, boy, girl).",
    "color": "Valid examples include: black, white, blue, red, green, yellow, navy, grey, brown, pink, beige.",
    "style": "Valid examples include: sporty, elegant, vintage, streetwear, casual, classic, minimalist, bohemian."
}

def normalize_field_value(field: str, val: str) -> str:
    """Normalize common synonyms for better FAISS filtering."""
    val = val.lower().strip()

    replacements = {
        "occasion": {
            "party outfit": "party", "birthday outfit": "birthday", "work": "office",
            "meeting": "office", "gym wear": "gym", "date night": "date", "friend's house": "social"
        },
        "formality": {
            "informal": "casual", "semi formal": "semi-formal", "business wear": "business",
            "smart": "semi-formal", "relaxed": "casual", "no rules": "casual"
        },
        "gender": {
            "girl": "female", "woman": "female", "lady": "female", "boy": "male",
            "man": "male", "gentleman": "male", "anyone": "unisex", "everyone": "unisex",
            "gender-neutral": "unisex"
        },
        "category": {
            "t-shirt": "tshirts", "shirt": "shirts", "shoe": "shoes", "sneakers": "sports shoes",
            "jeans": "jeans", "pants": "trousers", "hand bag": "handbags", "watch": "watches",
            "frock": "dresses", "gown": "dresses" # Example
        },
        "style": {
            "modern": "modern casual", "classy": "elegant", "athletic": "sporty",
            "retro": "vintage", "trim look": "elegant", "sharp": "elegant"
        },
        "color": {
            "dark": "black", "light": "white", "dark blue": "navy"
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


# INTENT HELPER

async def begin_intent_flow(user_query: str):
    """
    Reusable helper function to start an intent session and pre-fill.
    This version has a smarter prompt to help gemma3:1b infer fields.
    """
    if not user_query:
        return {"error": "Empty query."}

    # Step 1: Create a new session
    session_id, session = intent_manager.start(user_query)
    log_event("session_start", f"Session={session_id} | Query='{user_query}'")

    # Step 2: Try to pre-fill the intent with high-clarity extraction prompt
    extract_prompt = (
        "Task: You are a strict fashion entity extraction engine. Extract attributes from the user's outfit request into valid JSON.\n\n"
        "Attributes to extract:\n"
        f"- occasion ({FIELD_DEFINITIONS['occasion']})\n"
        f"- category ({FIELD_DEFINITIONS['category']})\n"
        f"- formality ({FIELD_DEFINITIONS['formality']})\n"
        f"- gender ({FIELD_DEFINITIONS['gender']})\n"
        f"- color ({FIELD_DEFINITIONS['color']})\n"
        f"- style ({FIELD_DEFINITIONS['style']})\n\n"
        "Rules:\n"
        "1. Extract ONLY attributes explicitly stated or clearly implied by the user.\n"
        "2. If an attribute is NOT mentioned, set its value to null.\n"
        "3. Output ONLY a valid JSON object with keys: \"occasion\", \"category\", \"formality\", \"gender\", \"color\", \"style\".\n"
        "4. Do NOT output markdown code blocks or explanations.\n\n"
        "Examples:\n"
        "Request: \"i need a black suit for a wedding\"\n"
        "JSON: {\"occasion\": \"wedding\", \"category\": \"suit\", \"formality\": \"formal\", \"gender\": \"male\", \"color\": \"black\", \"style\": \"classic\"}\n\n"
        "Request: \"looking for running shoes for gym\"\n"
        "JSON: {\"occasion\": \"gym\", \"category\": \"shoes\", \"formality\": \"casual\", \"gender\": null, \"color\": null, \"style\": \"sporty\"}\n\n"
        "Request: \"summer dress in red\"\n"
        "JSON: {\"occasion\": \"casual\", \"category\": \"dress\", \"formality\": \"casual\", \"gender\": \"female\", \"color\": \"red\", \"style\": null}\n\n"
        f"Request: \"{user_query}\"\n"
        "JSON:"
    )
    
    raw_res = "{}" # Default
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": extract_prompt, "stream": False, "format": "json"},
            timeout=60
        )
        r.raise_for_status()
        
        raw_res = r.json().get("response", "{}").strip()
        if "```" in raw_res:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_res, re.DOTALL)
            if m:
                raw_res = m.group(1)
        if "{" in raw_res and "}" in raw_res:
            raw_res = raw_res[raw_res.find("{"):raw_res.rfind("}")+1]

        extracted_data = json.loads(raw_res) if raw_res else {}

        # Step 3: Fill the session with extracted values
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

    # Step 4: Find the first missing field
    current_field = session["current_field"]

    if not current_field:
        log_event("intent_complete", f"Session={session_id} | Completed via pre-fill | Final={session['intent']}")
        return {"complete": True, "intent": session["intent"], "session_id": session_id, "question": "Thanks! I've got everything I need!"}

    # Step 5: Ask the first question with clear scope and examples
    intent_context = ", ".join([f"{k}: {v}" for k, v in session['intent'].items() if v])
    if not intent_context: intent_context = "(None yet)"
    
    ask_prompt = (
        "Role: You are Psyduck, a friendly and concise fashion advisor.\n"
        f"Context: The user wants an outfit. Known details: {intent_context}.\n"
        f"Missing Attribute to Ask: '{current_field}' ({FIELD_DEFINITIONS.get(current_field, '')}).\n"
        f"Attribute Guide: {FIELD_HINTS.get(current_field, '')}\n\n"
        "Instructions:\n"
        f"1. Ask exactly ONE clear, short question (under 15 words) asking for their preference for '{current_field}'.\n"
        "2. Include 2 quick examples in the question to help guide their answer.\n"
        "3. Output ONLY the question sentence. No greetings, quotes, or preambles.\n\n"
        "Question:"
    )
    
    q = f"What {current_field} are you looking for?" # Default
    try:
        r_q = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": ask_prompt, "stream": False},
            timeout=60
        )
        r_q.raise_for_status()
        q = (r_q.json().get("response") or q).strip().splitlines()[0].strip().strip('"')
        if not q: q = f"What about the {current_field}?"

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

# ROUTES

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ---- Chat (with Triage v2) ----
class ChatInput(BaseModel):
    text: str
    conversation_id: str | None = None

@app.post("/chat")
async def chat(data: ChatInput):
    conv_id = data.conversation_id or str(uuid.uuid4())
    user_text = data.text.strip()
    if not user_text:
        return {"reply": "Please enter a message.", "conversation_id": conv_id}

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

    # Step 1: Triage (classify message with strict categories)
    triage_prompt = (
        "Task: Determine if the user's message is a fashion outfit request or general conversational chatter.\n\n"
        f"Recent History:\n{context_str}\n\n"
        f"User Message: \"{user_text}\"\n\n"
        "Categories:\n"
        "- NEW_REQUEST: The user is requesting an outfit, asking for clothes, requesting recommendations, or specifying/changing style attributes (e.g. 'i need clothes', 'show suits', 'what about in blue?', 'black shoes').\n"
        "- CHAT: General pleasantries, jokes, greetings, or non-clothing talk (e.g. 'hi', 'thank you', 'how are you', 'tell me a joke', 'ok cool').\n\n"
        "Instructions: Output exactly ONE word: NEW_REQUEST or CHAT. Do not write explanation or punctuation.\n"
        "Classification:"
    )
    
    is_new_request = False
    try:
        r_triage = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": triage_prompt, "stream": False},
            timeout=60
        )
        r_triage.raise_for_status()
        res_text = (r_triage.json().get("response") or "CHAT").strip().upper()
        
        # Robust classification check with keyword fallback
        fashion_keywords = ["need", "want", "wear", "outfit", "dress", "suit", "shoes", "shirt", "pants", "jeans", "style", "recommend", "looking for"]
        if "NEW_REQUEST" in res_text or any(k in user_text.lower() for k in fashion_keywords):
            is_new_request = True
        else:
            is_new_request = False
            
        log_event("chat_triage", f"Conv={conv_id} | Text='{user_text}' | Triage='{res_text}' | Resolved={is_new_request}")
    except Exception as e:
        log_event("chat_triage_error", f"Conv={conv_id} | Error={e}")
        is_new_request = False

    # ... (rest of the /chat function) ...
    
    # Step 2: Handle based on Triage
    
    # --- PATH A: It's a NEW outfit request -> Start intent flow ---
    if is_new_request:
        log_event("chat_to_intent", f"Conv={conv_id} | Starting new intent flow for: '{user_text}'")
        
        intent_data = await begin_intent_flow(user_query=user_text)
        
        if "error" in intent_data:
             return {"reply": f"Error starting intent flow: {intent_data['error']}", "conversation_id": conv_id}
        
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
        res = requests.post(OLLAMA_URL, json={"model": get_ollama_model(), "prompt": prompt, "stream": False}, timeout=60)
        res.raise_for_status() # Fixed typo
        reply = parse_ollama_response(res.text) or "Hmm… Psyduck’s having a tiny brain fog."
        append_message(conv_id, "bot", reply)
        
        return {"reply": reply, "conversation_id": conv_id}
        
    except Exception as e:
        return {"reply": f"Error: {e}", "conversation_id": conv_id}


# INTENT START

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


# INTENT ANSWER

@app.post("/intent/answer")
async def intent_answer(body: dict):
    """
    Accept the user's answer for the current field of a session.
    This version uses "few-shot" triage and a creative question prompt.
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

    # Step 1: Triage - Is this an answer or a question?
    triage_prompt = (
        "Task: Classify whether the user is asking a QUESTION about what a fashion field means, or providing an ANSWER.\n\n"
        f"Context: The assistant asked for '{current_field}' ({FIELD_DEFINITIONS.get(current_field, '')}).\n"
        f"User Response: \"{answer}\"\n\n"
        "Classifications:\n"
        "- QUESTION: The user asks what the field means, asks for help/explanation, or asks why (e.g. 'what does that mean?', 'what is formality?', 'explain', 'huh?').\n"
        "- ANSWER: The user gives a preference, value, or choice (e.g. 'formal', 'male', 'red', 'casual', 'birthday', 'idk', 'any').\n\n"
        "Rules:\n"
        "1. Output exactly ONE word: QUESTION or ANSWER.\n"
        "2. Do not write punctuation or explanation.\n\n"
        "Classification:"
    )
    
    triage_result = "ANSWER"
    try:
        r_triage = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": triage_prompt, "stream": False},
            timeout=60
        )
        r_triage.raise_for_status()
        res_text = (r_triage.json().get("response") or "ANSWER").strip().upper()
        question_signals = ["?", "what is", "what does", "meaning", "explain", "why", "what do you mean", "huh", "what's"]
        if "QUESTION" in res_text or any(sig in answer.lower() for sig in question_signals):
            triage_result = "QUESTION"
        else:
            triage_result = "ANSWER"
        log_event("triage", f"Session={session_id} | Field={current_field} | Answer='{answer}' | Triage='{triage_result}'")
    except Exception as e:
        log_event("triage_error", f"Session={session_id} | Field={current_field} | Error={e}")
        triage_result = "ANSWER"

    # Step 2: Handle based on Triage
    
    # --- PATH A: User is asking a QUESTION ---
    if triage_result == "QUESTION":
        log_event("clarification", f"Session={session_id} | Field='{current_field}' | User is confused: '{answer}'")
        clarify_prompt = (
            "Role: You are Psyduck, a concise fashion assistant.\n"
            f"The user is asking about the outfit attribute '{current_field}'.\n"
            f"Definition: {FIELD_DEFINITIONS.get(current_field, 'a detail about your outfit')}.\n"
            f"Guide/Examples: {FIELD_HINTS.get(current_field, '')}\n"
            f"User Question: \"{answer}\"\n\n"
            "Instructions:\n"
            f"1. Explain what '{current_field}' means in 1 brief sentence using the definition.\n"
            f"2. Re-ask what they prefer for '{current_field}' in 1 short sentence offering 2 examples.\n"
            "3. Keep entire response under 30 words. No greetings or filler.\n\n"
            "Response:"
        )
        try:
            r2 = requests.post(
                OLLAMA_URL,
                json={"model": get_ollama_model(), "prompt": clarify_prompt, "stream": False},
                timeout=60
            )
            r2.raise_for_status()
            next_q = (r2.json().get("response") or "My mistake!").strip()
        except Exception as e:
            log_event("clarify_error", f"Session={session_id} | Field={current_field} | Error={e}")
            next_q = f"{current_field.capitalize()} refers to {FIELD_DEFINITIONS.get(current_field, 'your preference')}. What {current_field} would you prefer?"
        return {
            "complete": False, "next_field": current_field, "question": next_q,
            "intent": session["intent"], "note": "clarification_provided"
        }

    # --- PATH B: User gave an ANSWER ---
    interpret_prompt = (
        f"Task: Extract the user's value for the fashion attribute '{current_field}'.\n\n"
        f"Attribute: '{current_field}' ({FIELD_DEFINITIONS.get(current_field, '')})\n"
        f"Valid Guide: {FIELD_HINTS.get(current_field, '')}\n"
        f"User Input: \"{answer}\"\n\n"
        "Instructions:\n"
        f"1. Extract ONLY the single concise value for '{current_field}' from the user's input.\n"
        f"2. If the user input does NOT provide a valid value for '{current_field}', or says 'no', 'yes', 'idk', or is irrelevant, output exactly: UNKNOWN\n"
        "3. Output ONLY the extracted value or UNKNOWN. Do not add quotes, markdown, or explanation.\n\n"
        "Extracted Value:"
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": interpret_prompt, "stream": False},
            timeout=60
        )
        r.raise_for_status()
        val = (r.json().get("response") or "UNKNOWN").strip().strip('"\'.,!?').lower()
        invalids = ["unknown", "i don't know", "not sure", "none", "no idea", "maybe", "idk", "", "yes", "ok", "okay", "sure", "no", "nah", "nope"]
        if val in invalids or len(val.split()) > 4:
            val = "UNKNOWN"

        # Deterministic domain checks to prevent small model hallucinations
        if current_field == "gender" and val not in ["male", "female", "unisex", "man", "men", "woman", "women", "boy", "girl"]:
            val = "UNKNOWN"
        if current_field == "formality" and val not in ["casual", "semi-formal", "formal", "business"]:
            for f_val in ["casual", "semi-formal", "formal", "business"]:
                if f_val in val:
                    val = f_val
                    break
            else:
                val = "UNKNOWN"

        log_event("extract", f"Session={session_id} | Field={current_field} | Answer='{answer}' | Extracted='{val}'")
    except Exception as e:
        log_event("extract_error", f"Session={session_id} | Field={current_field} | Error={e}")
        val = "UNKNOWN"

    if val.upper() == "UNKNOWN":
        log_event("extract_failed", f"Session={session_id} | Field='{current_field}' | Answer='{answer}'")
        next_q = f"Could you clarify your {current_field}? ({FIELD_HINTS.get(current_field, FIELD_DEFINITIONS.get(current_field))})"
        return {
            "complete": False, "next_field": current_field, "question": next_q,
            "intent": session["intent"], "note": "extraction_failed"
        }

    # Step 3: Normalize and set the field
    val = normalize_field_value(current_field, val)
    intent_manager.set_field(session_id, current_field, val)
    session = intent_manager.get(session_id) # Get updated session
    next_field = session["current_field"]

    if not next_field:
        final_intent = session["intent"]
        log_event("intent_complete", f"Session={session_id} | Final={final_intent}")
        return {"complete": True, "intent": final_intent, "session_id": session_id}

    # Step 4: Ask model to generate next field question
    
    # Get the current intent state for context
    intent_context = ", ".join([f"{k}: {v}" for k, v in session['intent'].items() if v])
    if not intent_context: intent_context = "(No details yet)"

    ask_prompt = (
        "Role: You are Psyduck, a professional and concise fashion assistant.\n"
        f"Context: Known outfit preferences: {intent_context}.\n"
        f"Next Attribute to Ask: '{next_field}' ({FIELD_DEFINITIONS.get(next_field, '')}).\n"
        f"Attribute Guide: {FIELD_HINTS.get(next_field, '')}\n\n"
        "Instructions:\n"
        f"1. Ask exactly ONE short, professional question (under 15 words) asking for the user's preference for '{next_field}'.\n"
        "2. Provide 2 quick examples in the question to assist the user.\n"
        "3. Output ONLY the question. Do not include quotes, emojis, or introductory remarks.\n\n"
        "Question:"
    )

    try:
        r3 = requests.post(
            OLLAMA_URL,
            json={"model": get_ollama_model(), "prompt": ask_prompt, "stream": False},
            timeout=60
        )
        r3.raise_for_status()
        next_q = (r3.json().get("response") or f"Could you provide the {next_field}?").strip().splitlines()[0].strip().strip('"')
        if not next_q: next_q = f"What about the {next_field}?"
    except Exception as e:
        log_event("next_field_error", f"Session={session_id} | Field={next_field} | Error={e}")
        next_q = f"What about the {next_field}?"

    return {
        "complete": False,
        "next_field": next_field,
        "question": next_q,
        "intent": session["intent"]
    }


# RECOMMEND

@app.post("/recommend", response_class=JSONResponse)
async def recommend(data: dict):
    query = (data.get("query") or "").strip()
    intent = data.get("intent") or {}
    if not query:
        return JSONResponse({"error": "Empty query."}, status_code=400)

    if not os.path.exists(DB_PATH) or index is None:
        return JSONResponse({"error": "Database or FAISS index not ready. Please run setup.sh to build the database."}, status_code=503)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Step 1: Filter (SQL) ---
    filters, params = [], []
    for k, v in intent.items():
        if v:
            filters.append("(description LIKE ? OR category LIKE ?)")
            params += [f"%{v}%", f"%{v}%"]

    where_clause = " OR ".join(filters) if filters else "1=1"
    
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
    
    # --- NEW: Build a smarter query from the intent ---
    query_parts = [query] # Start with the original query
    
    # Add all the details we learned
    if intent.get("style"):
        query_parts.append(intent["style"])
    if intent.get("color"):
        query_parts.append(intent["color"])
    if intent.get("category"):
        query_parts.append(intent["category"])
    if intent.get("occasion"):
        query_parts.append(f"for a {intent['occasion']} event")
    if intent.get("formality"):
        query_parts.append(intent["formality"])
    
    final_query = " ".join(query_parts)
    log_event("recommend_query", f"Original: '{query}' | Final: '{final_query}'")
    
    # Encode the new, perfect query
    qv = model.encode([final_query]).astype("float32")
    # --- END NEW LOGIC ---

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
        return PlainTextResponse("No logs recorded yet.", status_code=200)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return PlainTextResponse(content[-8000:] or "Log is empty.", status_code=200)
    except Exception as e:
        return PlainTextResponse(f"Error reading logs: {e}", status_code=500)

# ---- Health ----
@app.get("/health")
async def health():
    return {"status": "ok", "model": get_ollama_model()}


# RUN

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Psyduck Outfit Assistant Server")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Ollama model name (e.g. llama3, qwen2.5-coder:3b, gemma3:1b, gpt-oss:120b-cloud)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    selected_model = args.model or os.getenv("OLLAMA_MODEL") or OLLAMA_MODEL
    os.environ["OLLAMA_MODEL"] = selected_model
    OLLAMA_MODEL = selected_model

    print(f"[INFO] Psyduck server listening at http://{args.host}:{args.port}")
    print(f"[INFO] Active Ollama model: '{selected_model}'")
    uvicorn.run("app:app", host=args.host, port=args.port, app_dir=BASE_DIR, reload=args.reload)