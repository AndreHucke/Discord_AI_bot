import asyncio
import logging
import os
import tempfile
import threading
import time
import wave
from pathlib import Path

import discord
import google.generativeai as genai
import numpy as np
import pyaudiowpatch as pyaudio
import whisper
from discord.ext import commands
from dotenv import load_dotenv
from gtts import gTTS

logging.basicConfig(level=logging.INFO)
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Missing required environment variables.")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")
stt_model = whisper.load_model(WHISPER_MODEL_SIZE)

# Audio Capture Tuning Parameters
AUDIO_THRESHOLD = 20      # Increase if it picks up background noise; decrease if it misses quiet speech
SILENCE_LIMIT = 2.0        # Seconds of silence required before processing the audio chunk
MAX_RECORD_SECONDS = 5.0
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# Shared async queue and control flags
audio_queue = asyncio.Queue()
stop_listening_event = threading.Event()

async def generate_reply(prompt: str) -> str:
    safe_prompt = (
        "Você é a 'segunda voz' caótica na mente de um personagem de RPG. "
        "O texto a seguir é uma transcrição de áudio de uma sessão de RPG ao vivo. "
        "Determine se este é um momento bom, dramático ou inapropriadamente engraçado para intervir. "
        "Se quiser falar, forneça uma resposta curta (menos de 2 frases) incorporando o personagem. "
        "Se preferir ficar quieto, responda exatamente com a palavra 'IGNORE'.\n\n"
        f"Os jogadores disseram: {prompt}"
    )
    response = await asyncio.to_thread(llm.generate_content, safe_prompt)
    return (response.text or "").strip()

async def transcribe_wav(wav_path: str) -> str:
    # Adding language="pt" makes it significantly more accurate for PT-BR
    result = await asyncio.to_thread(stt_model.transcribe, wav_path, language="pt")
    return result.get("text", "").strip()

async def synthesize_tts(text: str, output_mp3: str) -> None:
    def _save() -> None:
        tts = gTTS(text=text, lang="pt", tld="com.br")
        tts.save(output_mp3)
    await asyncio.to_thread(_save)

def loopback_listener_thread(loop: asyncio.AbstractEventLoop):
    p = pyaudio.PyAudio()
    
    try:
        target_id = 13 
        device_info = p.get_device_info_by_index(target_id)
        rate = int(device_info["defaultSampleRate"])
        channels = device_info["maxInputChannels"]
        
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=device_info["index"]
        )
        
        logging.info("SUCCESS: Monitoring system audio.")
        
        frames = []
        is_recording = False
        silence_start = None
        record_start_time = None # New safety timer
        
        while not stop_listening_event.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms > AUDIO_THRESHOLD:
                if not is_recording:
                    logging.info("🎤 Talking detected...")
                    record_start_time = time.time()
                is_recording = True
                silence_start = None
                frames.append(data)
            elif is_recording:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()

            # TRIGGER LOGIC: Process if silence is found OR if we've been recording too long
            if is_recording:
                current_time = time.time()
                silence_duration = (current_time - silence_start) if silence_start else 0
                total_duration = (current_time - record_start_time) if record_start_time else 0

                if silence_duration > SILENCE_LIMIT or total_duration > MAX_RECORD_SECONDS:
                    logging.info(f"⏹️ Triggered: Silence({silence_duration:.1f}s) or Max Time({total_duration:.1f}s)")
                    
                    filename = os.path.join(tempfile.gettempdir(), f"rpg_{int(time.time())}.wav")
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(rate)
                        wf.writeframes(b''.join(frames))
                    
                    loop.call_soon_threadsafe(audio_queue.put_nowait, filename)
                    
                    # Reset
                    frames = []
                    is_recording = False
                    silence_start = None
                    record_start_time = None
                    
    except Exception as e:
        logging.error(f"Loopback error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def process_audio_queue():
    while True:
        wav_path = await audio_queue.get()
        
        try:
            transcript = await transcribe_wav(wav_path)
            if not transcript or len(transcript) < 5:
                continue

            logging.info(f"Transcribed: {transcript}")
            
            reply_text = await generate_reply(transcript)
            
            # Respect the AI's decision to stay silent
            if not reply_text or "IGNORE" in reply_text.upper():
                logging.info("AI chose to remain silent.")
                continue

            logging.info(f"AI Interjection: {reply_text}")

            voice_client = bot.voice_clients[0] if bot.voice_clients else None
            if voice_client and voice_client.is_connected():
                
                tts_mp3 = os.path.join(tempfile.gettempdir(), "reply.mp3")
                await synthesize_tts(reply_text, tts_mp3)

                while voice_client.is_playing():
                    await asyncio.sleep(0.5)

                audio_source = discord.FFmpegPCMAudio(tts_mp3)
                voice_client.play(audio_source)
                
        except Exception as e:
            logging.error(f"Processing error: {e}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            audio_queue.task_done()

@bot.command(name="join")
async def join(ctx: commands.Context) -> None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Join a voice channel first, then run !join.")
        return

    try:
        if ctx.voice_client and ctx.voice_client.channel != ctx.author.voice.channel:
            await ctx.voice_client.move_to(ctx.author.voice.channel)
        else:
            await ctx.author.voice.channel.connect(timeout=15.0) 
        
        await ctx.send("Connected. Ready to speak.")
    except Exception as e:
        await ctx.send(f"Failed to connect: {e}")

@bot.command(name="listen")
async def listen(ctx: commands.Context) -> None:
    if not stop_listening_event.is_set() and threading.active_count() > 1:
        await ctx.send("Already actively monitoring audio.")
        return

    stop_listening_event.clear()
    loop = asyncio.get_running_loop()
    threading.Thread(target=loopback_listener_thread, args=(loop,), daemon=True).start()
    
    await ctx.send("Loopback capture started. The AI is now monitoring the session.")

@bot.command(name="stop")
async def stop(ctx: commands.Context) -> None:
    stop_listening_event.set()
    await ctx.send("Loopback capture stopped. The AI is no longer listening.")

@bot.command(name="test")
async def test(ctx: commands.Context) -> None:
    if not ctx.voice_client:
        await ctx.send("I need to be in a voice channel first! Run !join.")
        return

    await ctx.send("Generating test audio...")
    
    # A fun RPG-themed test phrase
    test_phrase = "Testando caraio"
    
    # Create the temporary MP3 file
    tts_mp3 = os.path.join(tempfile.gettempdir(), "test_audio.mp3")
    
    try:
        # Generate the audio using your existing TTS function
        await synthesize_tts(test_phrase, tts_mp3)

        # Wait if the bot is already playing something else
        while ctx.voice_client.is_playing():
            await asyncio.sleep(0.5)

        # Play the audio in Discord
        audio_source = discord.FFmpegPCMAudio(tts_mp3)
        ctx.voice_client.play(audio_source)
        
        await ctx.send("Speaking in the voice channel right now!")
        
    except Exception as e:
        await ctx.send(f"Error during test playback: {e}")

@bot.command(name="simulate")
async def simulate(ctx: commands.Context, *, scenario: str = None) -> None:
    if not ctx.voice_client:
        await ctx.send("I need to be in a voice channel first! Run !join.")
        return

    # Default RPG scenario if you don't provide one
    if not scenario:
        scenario = "Player 1: I want to try and jump across the chasm. Player 2: You have a strength stat of 8. You are going to die. Player 1: I roll for athletics anyway!"

    await ctx.send(f"🎙️ **Simulated Transcript:**\n> {scenario}")
    
    # 1. Feed the text to Gemini
    reply_text = await generate_reply(scenario)
    
    # 2. Check if the AI chose to stay silent
    if not reply_text or "IGNORE" in reply_text.upper():
        await ctx.send("🤖 **AI Decision:** `IGNORE` (Decided to stay silent)")
        return
        
    await ctx.send(f"🗣️ **AI Interjection:**\n> {reply_text}")

    # 3. Generate text-to-speech and play it
    tts_mp3 = os.path.join(tempfile.gettempdir(), "simulate_audio.mp3")
    try:
        await synthesize_tts(reply_text, tts_mp3)

        while ctx.voice_client.is_playing():
            await asyncio.sleep(0.5)

        audio_source = discord.FFmpegPCMAudio(tts_mp3)
        ctx.voice_client.play(audio_source)
        
    except Exception as e:
        await ctx.send(f"Error during playback: {e}")

@bot.command(name="debug_mic")
async def debug_mic(ctx: commands.Context) -> None:
    await ctx.send("🎤 **FORCE CAPTURE STARTING...** Speak now for 5 seconds!")
    
    # Recording settings
    target_id = 13 
    p = pyaudio.PyAudio()
    dev_info = p.get_device_info_by_index(target_id)
    rate = int(dev_info["defaultSampleRate"])
    channels = dev_info["maxInputChannels"]
    
    frames = []
    
    def record():
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, input_device_index=target_id)
        for _ in range(0, int(rate / 1024 * 5)): # Record exactly 5 seconds
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()

    # Run recording in a thread so we don't freeze the bot
    await asyncio.to_thread(record)
    await ctx.send("💾 **Capture finished. Transcribing...**")

    # Save and Transcribe
    temp_path = os.path.join(tempfile.gettempdir(), "debug_direct.wav")
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    
    p.terminate()

    transcript = await transcribe_wav(temp_path)
    
    if transcript:
        await ctx.send(f"✅ **Whisper heard:** \"{transcript}\"")
    else:
        await ctx.send("❌ **Whisper heard nothing.** (If volume numbers were moving, check if Whisper is loaded correctly)")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

async def finished_callback_debug(sink, channel: discord.TextChannel, *args):
    await channel.send("💾 **Processing your voice...**")
    
    for user_id, audio in sink.audio_data.items():
        # Save the captured audio to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio.file.getvalue())
            temp_path = f.name
        
        # Run Whisper STT
        result = stt_model.transcribe(temp_path)
        transcript = result['text'].strip()
        
        if transcript:
            await channel.send(f"✅ **Whisper heard:** \"{transcript}\"")
        else:
            await channel.send("❌ **Whisper heard nothing.** (Check your mic settings/permissions)")
            
        os.remove(temp_path)

@bot.command(name="leave")
async def leave(ctx: commands.Context) -> None:
    stop_listening_event.set()
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected.")

@bot.event
async def on_ready() -> None:
    logging.info(f"Bot logged in as {bot.user}")
    bot.loop.create_task(process_audio_queue())

bot.run(DISCORD_TOKEN)