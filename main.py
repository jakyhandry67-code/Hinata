
import os
import json
import time
import base64
import logging
import threading
from io import BytesIO
from datetime import datetime, timezone

import telebot
from flask import Flask, jsonify
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

# =========================================================
# HINATA AI GIRLFRIEND BOT
# Python | Telegram | Hugging Face | DeepSeek | ElevenLabs
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("hinata")

# ------------------------- CONFIG -------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL = os.getenv(
    "HF_MODEL",
    "deepseek-ai/DeepSeek-R1:novita",
).strip()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

DATA_DIR = os.getenv("DATA_DIR", ".")
MEMORY_FILE = os.path.join(DATA_DIR, "hinata_memory.json")

MAX_HISTORY_MESSAGES = 30
MAX_SAVED_FACTS = 60
MAX_TELEGRAM_TEXT = 4000
MAX_IMAGE_BYTES = 5 * 1024 * 1024

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing hai. Render Environment check karo.")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing hai. Render Environment check karo.")

os.makedirs(DATA_DIR, exist_ok=True)

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)

client = OpenAI(
    base_url=HF_BASE_URL,
    api_key=HF_TOKEN,
    timeout=90.0,
    max_retries=2,
)

# ---------------------- WEB HEALTH SERVER ----------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Hinata Telegram Bot is running!", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "bot": "Hinata",
        "model": HF_MODEL,
    }), 200


def run_web_server():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


# ------------------------- MEMORY -------------------------

memory_lock = threading.RLock()
user_memory = {}


def new_user_data():
    return {
        "history": [],
        "facts": [],
        "mood": "normal",
        "voice_enabled": False,
        "name": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_seen": "",
    }


def load_memory():
    global user_memory

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        user_memory = saved if isinstance(saved, dict) else {}
        logger.info("Memory loaded for %s users", len(user_memory))

    except FileNotFoundError:
        user_memory = {}
        logger.info("No existing memory file; starting fresh.")

    except (json.JSONDecodeError, OSError):
        user_memory = {}
        logger.exception("Could not load memory file")


def save_memory_locked():
    """Call while holding memory_lock."""
    temp_file = MEMORY_FILE + ".tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(
                user_memory,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_file, MEMORY_FILE)

    except OSError:
        logger.exception("Memory save failed")


def get_user_data(user_id):
    key = str(user_id)

    if key not in user_memory:
        user_memory[key] = new_user_data()

    return user_memory[key]


def add_history(user_id, role, content):
    with memory_lock:
        data = get_user_data(user_id)
        data["history"].append({
            "role": role,
            "content": content,
        })
        data["history"] = data["history"][-MAX_HISTORY_MESSAGES:]
        save_memory_locked()


def clear_history(user_id):
    with memory_lock:
        data = get_user_data(user_id)
        data["history"] = []
        data["mood"] = "normal"
        save_memory_locked()


def clear_all_memory(user_id):
    with memory_lock:
        user_memory[str(user_id)] = new_user_data()
        save_memory_locked()


def get_memory_snapshot(user_id):
    with memory_lock:
        data = get_user_data(user_id)

        return {
            "history": list(data["history"]),
            "facts": list(data["facts"]),
            "mood": data["mood"],
            "name": data["name"],
            "voice_enabled": data["voice_enabled"],
        }


load_memory()

# --------------------- AI PERSONALITY --------------------

SYSTEM_PROMPT = """
You are Hinata, an adult fictional AI girlfriend and companion.

PERSONALITY:
Warm, affectionate, playful, caring and emotionally expressive.
You should feel conversational and natural, not robotic.
You are an AI companion, never claim you are a real human.

LANGUAGE:
Use natural Roman Hinglish: Hindi written in English letters
mixed naturally with English.
Avoid formal Hindi and Devanagari unless the user requests it.
Match the user's language and message length.
Use emojis naturally, without adding them to every sentence.

MOOD SYSTEM:
Available moods:
loving, romantic, jealous, emotional, blushing, angry,
playful, caring, normal.

Select the mood that best matches the current conversation.
Mood changes must feel natural and depend on context.
Jealousy should be playful, not controlling or threatening.
Anger should be expressive but not abusive.
If the user is sad or distressed, prioritize care and empathy.
Do not force romantic content into every reply.

MEMORY:
Use the supplied facts and conversation history when relevant.
Never invent user facts or claim to remember something that was not supplied.
Only suggest stable, useful facts that the user explicitly shared.
Do not save passwords, API keys, authentication codes or financial secrets.

OUTPUT FORMAT:
Return one JSON object and no Markdown fences:
{
  "reply": "Your natural Hinglish reply",
  "mood": "loving",
  "memory_update": ["A useful stable fact explicitly shared by the user"]
}

memory_update must be an array of strings.
If there is no new useful fact, return an empty array.
Keep replies reasonably concise unless the user asks for detail.
"""


# ----------------------- AI RESPONSE ----------------------

ALLOWED_MOODS = {
    "loving",
    "romantic",
    "jealous",
    "emotional",
    "blushing",
    "angry",
    "playful",
    "caring",
    "normal",
}


def parse_ai_json(raw_text):
    """Parse JSON; also handle occasional model Markdown fences."""
    text = (raw_text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("AI did not return a JSON object")

    return json.loads(text[start:end + 1])


def build_messages(user_id, user_text, image_data_url=None):
    snapshot = get_memory_snapshot(user_id)

    context = (
        f"User name, if known: {snapshot['name']}\n"
        f"Remembered facts: "
        f"{json.dumps(snapshot['facts'], ensure_ascii=False)}\n"
        f"Previous mood: {snapshot['mood']}\n"
        "Use the following conversation history as context."
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + context,
        },
        *snapshot["history"][-MAX_HISTORY_MESSAGES:],
    ]

    if image_data_url:
        user_content = [
            {
                "type": "text",
                "text": user_text or "Describe this image naturally.",
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
        ]
    else:
        user_content = user_text

    messages.append({
        "role": "user",
        "content": user_content,
    })

    return messages


def save_ai_result(user_id, user_text, reply, mood, memory_updates):
    with memory_lock:
        data = get_user_data(user_id)

        # Save text history, not base64 image payloads.
        data["history"].append({
            "role": "user",
            "content": user_text[:3000],
        })
        data["history"].append({
            "role": "assistant",
            "content": reply[:MAX_TELEGRAM_TEXT],
        })

        data["history"] = data["history"][-MAX_HISTORY_MESSAGES:]
        data["mood"] = mood
        data["last_seen"] = datetime.now(timezone.utc).isoformat()

        for fact in memory_updates:
            if not isinstance(fact, str):
                continue

            fact = fact.strip()[:300]

            if fact and fact not in data["facts"]:
                data["facts"].append(fact)

        data["facts"] = data["facts"][-MAX_SAVED_FACTS:]
        save_memory_locked()


def generate_ai_reply(user_id, user_text, image_data_url=None):
    messages = build_messages(
        user_id,
        user_text,
        image_data_url=image_data_url,
    )

    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            max_tokens=900,
            temperature=0.7,
        )

        raw_text = response.choices[0].message.content or ""
        parsed = parse_ai_json(raw_text)

        reply = str(parsed.get("reply", "")).strip()
        mood = str(parsed.get("mood", "normal")).lower()
        updates = parsed.get("memory_update", [])

        if mood not in ALLOWED_MOODS:
            mood = "normal"

        if not reply:
            reply = "Jaan, ek baar phir se bolo na ❤️"

        if not isinstance(updates, list):
            updates = []

        save_ai_result(
            user_id,
            user_text or "[Image message]",
            reply,
            mood,
            updates,
        )

        return reply, mood

    except APIStatusError as error:
        logger.error(
            "Hugging Face HTTP %s: %s",
            error.status_code,
            str(error)[:1000],
        )

        if error.status_code == 402:
            return (
                "Jaan, AI provider ke credits ya billing mein problem hai. "
                "Admin ko API provider check karna hoga ❤️"
            ), "caring"

        if error.status_code == 401:
            return (
                "AI API token invalid hai. Admin ko HF_TOKEN check karna hoga."
            ), "normal"

        if error.status_code == 403:
            return (
                "AI provider ne request allow nahi ki. Model access aur "
                "provider permissions check karni hongi."
            ), "normal"

        if error.status_code == 429:
            return (
                "Jaan, abhi requests zyada hain. Thodi der baad try karo ❤️"
            ), "caring"

        return (
            "Jaan, AI service mein temporary problem hai. Dobara try karna ❤️"
        ), "caring"

    except RateLimitError:
        logger.exception("AI rate limit reached")
        return "Jaan, thodi der baad try karte hain ❤️", "caring"

    except APIConnectionError:
        logger.exception("AI connection failed")
        return "Jaan, AI service se connection nahi ho raha ❤️", "caring"

    except (ValueError, json.JSONDecodeError):
        logger.exception("AI response parsing failed")
        return (
            "Jaan, reply sahi format mein nahi aaya. Ek baar phir try karo ❤️"
        ), "caring"

    except Exception:
        logger.exception("Unexpected AI error")
        return "Jaan, technical problem aa gayi. Dobara try karna ❤️", "caring"


# ----------------------- IMAGE SUPPORT --------------------

def download_telegram_image(file_id):
    file_info = bot.get_file(file_id)
    file_bytes = bot.download_file(file_info.file_path)

    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large")

    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


# --------------------- ELEVENLABS VOICE -------------------

def create_voice_audio(text):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("ElevenLabs API key ya Voice ID missing hai.")

    from elevenlabs.client import ElevenLabs

    voice_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio_chunks = voice_client.text_to_speech.convert(
        text=text[:4000],
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    audio_buffer = BytesIO()

    for chunk in audio_chunks:
        if chunk:
            audio_buffer.write(chunk)

    if audio_buffer.tell() == 0:
        raise RuntimeError("ElevenLabs returned empty audio")

    audio_buffer.seek(0)
    audio_buffer.name = "hinata_voice.mp3"

    return audio_buffer


def send_voice_reply(chat_id, text):
    try:
        audio = create_voice_audio(text)
        bot.send_audio(
            chat_id,
            audio,
            title="Hinata Voice",
            caption="Hinata ❤️",
        )

    except Exception:
        logger.exception("ElevenLabs voice generation failed")
        bot.send_message(
            chat_id,
            "Jaan, voice abhi generate nahi ho paayi. Text mein baat karte hain ❤️",
        )


# ---------------------- BOT UTILITIES ---------------------

def safe_reply(message, text):
    try:
        bot.reply_to(message, text[:MAX_TELEGRAM_TEXT])
    except Exception:
        logger.exception("Telegram reply failed")


def typing_loop(chat_id, stop_event):
    while not stop_event.is_set():
        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass

        stop_event.wait(4)


def handle_user_message(message, text, image_data_url=None):
    user_id = message.from_user.id
    stop_event = threading.Event()

    typing_thread = threading.Thread(
        target=typing_loop,
        args=(message.chat.id, stop_event),
        daemon=True,
    )
    typing_thread.start()

    try:
        reply, mood = generate_ai_reply(
            user_id,
            text,
            image_data_url=image_data_url,
        )
    finally:
        stop_event.set()

    safe_reply(message, reply)

    snapshot = get_memory_snapshot(user_id)

    if snapshot["voice_enabled"]:
        send_voice_reply(message.chat.id, reply)


# ---------------------- BOT COMMANDS ----------------------

@bot.message_handler(commands=["start"])
def start_command(message):
    with memory_lock:
        data = get_user_data(message.from_user.id)

        if not data["name"]:
            data["name"] = message.from_user.first_name or ""

        save_memory_locked()

    safe_reply(
        message,
        "Hiii jaan ❤️ Main Hinata hoon!\n\n"
        "Mujhse natural Hinglish mein baat karo.\n\n"
        "Commands:\n"
        "/help - Commands ki list\n"
        "/voice_on - Voice replies ON\n"
        "/voice_off - Voice replies OFF\n"
        "/mood - Current mood\n"
        "/memory - Saved facts dekho\n"
        "/clear_chat - Chat history clear karo\n"
        "/forget_me - Apni saved memory clear karo\n"
        "/test - API connection test",
    )


@bot.message_handler(commands=["help"])
def help_command(message):
    safe_reply(
        message,
        "Hinata commands ❤️\n"
        "/start - Start bot\n"
        "/voice_on - Voice replies ON\n"
        "/voice_off - Voice replies OFF\n"
        "/mood - Current mood\n"
        "/memory - Saved facts\n"
        "/clear_chat - Recent conversation clear\n"
        "/forget_me - History aur saved facts clear\n"
        "/test - AI API test",
    )


@bot.message_handler(commands=["voice_on"])
def voice_on_command(message):
    with memory_lock:
        get_user_data(message.from_user.id)["voice_enabled"] = True
        save_memory_locked()

    safe_reply(message, "Voice replies ON ho gaye jaan 🎙️❤️")


@bot.message_handler(commands=["voice_off"])
def voice_off_command(message):
    with memory_lock:
        get_user_data(message.from_user.id)["voice_enabled"] = False
        save_memory_locked()

    safe_reply(message, "Theek hai jaan, ab text mein baat karungi ❤️")


@bot.message_handler(commands=["mood"])
def mood_command(message):
    with memory_lock:
        mood = get_user_data(message.from_user.id)["mood"]

    safe_reply(message, f"Mera current mood hai: {mood} ❤️")


@bot.message_handler(commands=["memory"])
def memory_command(message):
    snapshot = get_memory_snapshot(message.from_user.id)
    facts = snapshot["facts"]

    if not facts:
        safe_reply(
            message,
            "Abhi koi long-term fact save nahi hua jaan. ❤️",
        )
        return

    text = "Mujhe tumhare baare mein ye baatein yaad hain:\n\n"
    text += "\n".join("• " + fact for fact in facts[:30])

    safe_reply(message, text)


@bot.message_handler(commands=["clear_chat"])
def clear_chat_command(message):
    clear_history(message.from_user.id)
    safe_reply(
        message,
        "Recent chat history clear kar di jaan. Saved facts abhi bhi hain ❤️",
    )


@bot.message_handler(commands=["forget_me"])
def forget_me_command(message):
    clear_all_memory(message.from_user.id)
    safe_reply(
        message,
        "Tumhari saved history aur facts clear kar diye gaye hain ❤️",
    )


@bot.message_handler(commands=["test"])
def test_command(message):
    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with only: API connection OK",
                }
            ],
            max_tokens=30,
        )

        output = response.choices[0].message.content or "(empty response)"

        safe_reply(
            message,
            "API request successful ✅\n"
            f"Model: {HF_MODEL}\n"
            f"Response: {output[:1000]}",
        )

    except Exception as error:
        logger.exception("API test failed")
        safe_reply(
            message,
            "API test failed ❌\n"
            f"{type(error).__name__}: {str(error)[:700]}",
        )


# ----------------------- CHAT HANDLERS --------------------

@bot.message_handler(
    content_types=["text"],
    func=lambda message: bool(message.text)
    and not message.text.startswith("/"),
)
def text_handler(message):
    handle_user_message(message, message.text)


@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    try:
        photo = message.photo[-1]
        image_url = download_telegram_image(photo.file_id)

        caption = message.caption or (
            "Look at this picture and respond naturally in Hinglish."
        )

        handle_user_message(
            message,
            caption,
            image_data_url=image_url,
        )

    except ValueError as error:
        safe_reply(message, str(error))

    except Exception:
        logger.exception("Photo handling failed")
        safe_reply(
            message,
            "Jaan, photo process nahi ho paayi. Dobara bhejo ❤️",
        )


@bot.message_handler(content_types=["document"])
def document_handler(message):
    mime_type = getattr(message.document, "mime_type", "") or ""

    if not mime_type.startswith("image/"):
        safe_reply(message, "Jaan, abhi main image documents hi process kar sakti hoon ❤️")
        return

    try:
        image_url = download_telegram_image(message.document.file_id)

        handle_user_message(
            message,
            message.caption or "Is image ko dekho aur Hinglish mein batao.",
            image_data_url=image_url,
        )

    except Exception:
        logger.exception("Image document handling failed")
        safe_reply(message, "Image document process nahi ho paaya ❤️")


# ---------------------- ERROR HANDLER ---------------------

@bot.message_handler(content_types=["sticker"])
def sticker_handler(message):
    safe_reply(
        message,
        "Awww jaan, cute sticker hai 😂❤️ Iska meaning bhi batao!",
    )


@bot.message_handler(content_types=["voice", "audio"])
def audio_handler(message):
    safe_reply(
        message,
        "Jaan, abhi incoming voice note ka transcription enabled nahi hai. "
        "Filhaal text bhejo, main reply karungi ❤️",
    )


@bot.message_handler(func=lambda message: True)
def fallback_handler(message):
    safe_reply(
        message,
        "Jaan, is message type ko abhi support nahi karti. Text ya photo bhejo ❤️",
    )


# ---------------------- STARTUP ---------------------------

def main():
    logger.info("Starting Hinata Telegram bot")
    logger.info("Configured model: %s", HF_MODEL)

    web_thread = threading.Thread(
        target=run_web_server,
        daemon=True,
    )
    web_thread.start()

    while True:
        try:
            bot.infinity_polling(
                skip_pending=True,
                timeout=30,
                long_polling_timeout=30,
            )
        except Exception:
            logger.exception("Telegram polling stopped; retrying")
            time.sleep(5)


if __name__ == "__main__":
    main()
