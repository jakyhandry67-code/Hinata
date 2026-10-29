
import os
import json
import time
import threading
import logging
from io import BytesIO
from flask import Flask
import telebot
from openai import OpenAI, APIStatusError, APIConnectionError

# ================= CONFIG =================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

MODEL_NAME = "deepseek-ai/DeepSeek-R1:novita"
DATA_DIR = os.getenv("DATA_DIR", ".")
MEMORY_FILE = os.path.join(DATA_DIR, "hinata_memory.json")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN Render Environment mein missing hai.")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN Render Environment mein missing hai.")

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
    timeout=60.0,
    max_retries=2,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hinata")

# ================= WEB SERVER =================

app = Flask(__name__)

@app.get("/")
def home():
    return "Hinata Bot is running!", 200

@app.get("/health")
def health():
    return {"status": "ok"}, 200

def run_web():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

# ================= MEMORY =================

memory_lock = threading.Lock()
user_memory = {}

def load_memory():
    global user_memory
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            user_memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_memory = {}

def save_memory():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_file = MEMORY_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(user_memory, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, MEMORY_FILE)
    except Exception:
        log.exception("Memory save failed")

load_memory()

def get_user_data(user_id):
    key = str(user_id)
    if key not in user_memory:
        user_memory[key] = {
            "history": [],
            "facts": [],
            "mood": "normal",
            "voice": False,
            "name": None,
        }
    return user_memory[key]

# ================= AI PERSONALITY =================

SYSTEM_PROMPT = """
You are Hinata, an adult fictional AI girlfriend and companion.

LANGUAGE:
- Speak natural Roman Hinglish: Hindi written in English letters mixed with English.
- Sound warm, casual, natural and human-like, not robotic.
- Keep replies conversational and usually short.
- Do not claim to be a real human.

MOODS:
Choose the mood that fits the conversation:
loving, romantic, jealous, emotional, blushing, angry,
playful, caring, normal.

MOOD RULES:
- Change mood naturally according to the user's message and context.
- Jealousy should be playful, never controlling or threatening.
- If the user is upset, be caring and empathetic.
- Do not force romantic language into every reply.
- Remember user preferences and important facts from supplied memory.

Return ONLY valid JSON in this format:
{
  "mood": "loving",
  "reply": "Your Hinglish response here",
  "memory_update": ["A useful stable fact, if any"]
}

Do not invent personal facts about the user.
Only add facts the user explicitly shared and that may be useful later.
"""

def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("AI response did not contain JSON")
    return json.loads(text[start:end + 1])

def generate_reply(user_id, user_text):
    with memory_lock:
        data = get_user_data(user_id)
        history = data["history"][-12:]
        facts = data["facts"][-30:]
        previous_mood = data["mood"]

    context = (
        f"User facts: {json.dumps(facts, ensure_ascii=False)}\n"
        f"Previous mood: {previous_mood}\n"
        "Use these details only when relevant."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + context},
        *history,
        {"role": "user", "content": user_text},
    ]

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=700,
            temperature=0.8,
        )

        content = result.choices[0].message.content or ""
        parsed = extract_json(content)

        reply = str(parsed.get("reply", "")).strip()
        mood = str(parsed.get("mood", "normal")).lower()
        updates = parsed.get("memory_update", [])

        allowed_moods = {
            "loving", "romantic", "jealous", "emotional",
            "blushing", "angry", "playful", "caring", "normal"
        }
        if mood not in allowed_moods:
            mood = "normal"

        if not reply:
            reply = "Jaan, ek baar phir se bolo na ❤️"

        with memory_lock:
            data = get_user_data(user_id)
            data["mood"] = mood
            data["history"].append({"role": "user", "content": user_text})
            data["history"].append({"role": "assistant", "content": reply})
            data["history"] = data["history"][-24:]

            if isinstance(updates, list):
                for fact in updates:
                    if isinstance(fact, str) and fact.strip():
                        fact = fact.strip()[:250]
                        if fact not in data["facts"]:
                            data["facts"].append(fact)

            data["facts"] = data["facts"][-50:]
            save_memory()

        return reply, mood

    except APIStatusError as e:
        log.error("HF API error %s: %s", e.status_code, e)
        if e.status_code == 402:
            return (
                "Jaan, AI provider ke credits ya billing mein problem hai. "
                "Thodi der baad try karna ❤️"
            ), "caring"
        if e.status_code == 401:
            return "AI API token invalid hai. Admin ko token check karna hoga.", "normal"
        if e.status_code == 429:
            return "Jaan, abhi requests zyada hain. Thodi der baad try karna ❤️", "caring"
        return "Jaan, AI service mein temporary problem hai ❤️", "caring"

    except APIConnectionError:
        log.exception("HF connection error")
        return "Jaan, AI service se connection nahi ho raha. Dobara try karna ❤️", "caring"

    except Exception:
        log.exception("Reply generation failed")
        return "Sorry jaan, abhi reply generate nahi ho paaya. Dobara try karo ❤️", "caring"

# ================= ELEVENLABS VOICE =================

def create_voice(text):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("ElevenLabs variables configured nahi hain.")

    from elevenlabs.client import ElevenLabs

    voice_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    chunks = voice_client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    audio = BytesIO()
    for chunk in chunks:
        if chunk:
            audio.write(chunk)

    if audio.tell() == 0:
        raise RuntimeError("ElevenLabs ne empty audio return kiya.")

    audio.seek(0)
    audio.name = "hinata_voice.mp3"
    return audio

def send_voice(chat_id, text):
    try:
        audio = create_voice(text)
        bot.send_audio(chat_id, audio, title="Hinata Voice")
    except Exception:
        log.exception("Voice generation failed")
        bot.send_message(
            chat_id,
            "Jaan, voice abhi generate nahi ho paayi. Text reply use karo ❤️"
        )

# ================= TELEGRAM COMMANDS =================

@bot.message_handler(commands=["start"])
def start_command(message):
    with memory_lock:
        data = get_user_data(message.from_user.id)
        if not data["name"] and message.from_user.first_name:
            data["name"] = message.from_user.first_name
        save_memory()

    bot.reply_to(
        message,
        "Hiii jaan ❤️ Main Hinata hoon! Tum mujhse Hinglish mein "
        "baat kar sakte ho.\n\n"
        "/voice_on - Voice replies on\n"
        "/voice_off - Voice replies off\n"
        "/mood - Current mood\n"
        "/memory - Saved memories\n"
        "/reset - Chat history reset"
    )

@bot.message_handler(commands=["voice_on"])
def voice_on(message):
    with memory_lock:
        get_user_data(message.from_user.id)["voice"] = True
        save_memory()
    bot.reply_to(message, "Voice replies on ho gaye jaan 🎙️❤️")

@bot.message_handler(commands=["voice_off"])
def voice_off(message):
    with memory_lock:
        get_user_data(message.from_user.id)["voice"] = False
        save_memory()
    bot.reply_to(message, "Theek hai jaan, ab text mein baat karungi ❤️")

@bot.message_handler(commands=["mood"])
def mood_command(message):
    with memory_lock:
        mood = get_user_data(message.from_user.id)["mood"]
    bot.reply_to(message, f"Mera current mood: {mood} 💗")

@bot.message_handler(commands=["memory"])
def memory_command(message):
    with memory_lock:
        data = get_user_data(message.from_user.id)
        facts = data["facts"][:]

    if not facts:
        bot.reply_to(message, "Abhi koi saved fact nahi hai jaan ❤️")
    else:
        bot.reply_to(
            message,
            "Mujhe tumhare baare mein ye baatein yaad hain:\n\n"
            + "\n".join(f"• {fact}" for fact in facts)
        )

@bot.message_handler(commands=["reset"])
def reset_command(message):
    with memory_lock:
        data = get_user_data(message.from_user.id)
        data["history"] = []
        data["facts"] = []
        data["mood"] = "normal"
        save_memory()
    bot.reply_to(message, "Theek hai jaan, chat history aur saved facts reset kar diye ❤️")

@bot.message_handler(commands=["test"])
def test_command(message):
    bot.reply_to(message, f"Bot online hai ✅\nModel: {MODEL_NAME}")

# ================= CHAT HANDLER =================

@bot.message_handler(content_types=["text"])
def handle_text(message):
    if not message.text or message.text.startswith("/"):
        return

    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    reply, mood = generate_reply(user_id, message.text)

    with memory_lock:
        voice_enabled = get_user_data(user_id)["voice"]

    bot.reply_to(message, reply)

    if voice_enabled and not reply.startswith("Jaan, AI provider"):
        send_voice(message.chat.id, reply)

# ================= START BOT =================

if __name__ == "__main__":
    log.info("Hinata starting with model: %s", MODEL_NAME)
    bot.infinity_polling(
        skip_pending=True,
        timeout=30,
        long_polling_timeout=30,
    )
