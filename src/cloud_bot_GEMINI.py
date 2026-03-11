import asyncio
import logging
import os
import tempfile
import threading
import wave

import discord
import google.generativeai as genai
import pyaudiowpatch as pyaudio
import whisper
from discord.ext import commands
from dotenv import load_dotenv
import edge_tts

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

# Audio Capture Settings
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt32
LOOPBACK_DEVICE_ID = 13  # Your system audio device

# Recording state
recording_frames = []
recording_thread = None
stop_recording_event = threading.Event()
audio_settings = None


async def generate_reply(prompt: str) -> str:
    """Generate AI response to transcript."""
    safe_prompt = (
        "Você é uma voz divina que guia um druida que nao lembra nada do passado dele. Voce nem sempre eh claro e constantemente fala de forma enigimatica, divertida ou caotica. "
        "O texto a seguir é uma transcrição de áudio de uma sessão de RPG ao vivo. "
        "Forneça uma resposta curta e divertida (uma frase) incorporando essa voz.\n\n"
        f"Os jogadores disseram: {prompt}"
    )
    response = await asyncio.to_thread(llm.generate_content, safe_prompt)
    return (response.text or "").strip()


async def transcribe_wav(wav_path: str) -> str:
    """Transcribe audio file using Whisper."""
    result = await asyncio.to_thread(stt_model.transcribe, wav_path, language="pt")
    return result.get("text", "").strip()


async def synthesize_tts(text: str, output_mp3: str) -> None:
    # 'Francisca' is a great, clear PT-BR voice
    communicate = edge_tts.Communicate(text, "pt-BR-AntonioNeural")
    await communicate.save(output_mp3)


def capture_audio_thread():
    """Background thread that captures all audio until stopped."""
    global recording_frames, audio_settings
    
    p = pyaudio.PyAudio()
    
    try:
        device_info = p.get_device_info_by_index(LOOPBACK_DEVICE_ID)
        rate = int(device_info["defaultSampleRate"])
        channels = device_info["maxInputChannels"]
        
        # Store settings for saving later
        audio_settings = {
            "rate": rate,
            "channels": channels,
            "sample_width": p.get_sample_size(FORMAT)
        }
        
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=device_info["index"]
        )
        
        logging.info("🎤 Recording started...")
        
        # Continuously capture audio until stopped
        while not stop_recording_event.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            recording_frames.append(data)
                    
    except Exception as e:
        logging.error(f"Recording error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        logging.info("🎤 Recording stopped.")


@bot.command(name="join")
async def join(ctx: commands.Context) -> None:
    """Join the user's voice channel."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("⚠️ Join a voice channel first, then run !join.")
        return

    try:
        if ctx.voice_client and ctx.voice_client.channel != ctx.author.voice.channel:
            await ctx.voice_client.move_to(ctx.author.voice.channel)
        else:
            await ctx.author.voice.channel.connect(timeout=15.0) 
        
        await ctx.send("✅ Connected! Use `!start` to begin capturing audio.")
    except Exception as e:
        await ctx.send(f"❌ Failed to connect: {e}")


@bot.command(name="start")
async def start(ctx: commands.Context) -> None:
    """Start capturing audio from system loopback."""
    global recording_frames, recording_thread, stop_recording_event
    
    if recording_thread and recording_thread.is_alive():
        await ctx.send("⚠️ Already recording! Use `!stop` to finish.")
        return
    
    # Reset state
    recording_frames = []
    stop_recording_event.clear()
    
    # Start recording thread
    recording_thread = threading.Thread(target=capture_audio_thread, daemon=True)
    recording_thread.start()
    
    await ctx.send("🎙️ Recording started! Use `!stop` when you're done.")


@bot.command(name="stop")
async def stop(ctx: commands.Context) -> None:
    """Stop recording, transcribe, get AI response, and play it."""
    global recording_frames, recording_thread, audio_settings
    
    if not recording_thread or not recording_thread.is_alive():
        await ctx.send("⚠️ No recording in progress. Use `!start` first.")
        return
    
    # Signal the thread to stop
    stop_recording_event.set()
    await ctx.send("⏹️ Stopping recording and processing...")
    
    # Wait for thread to finish
    await asyncio.to_thread(recording_thread.join, timeout=2.0)
    
    # Check if we have audio data
    if not recording_frames or not audio_settings:
        await ctx.send("❌ No audio was captured.")
        return
    
    try:
        # Save the recording
        temp_wav = os.path.join(tempfile.gettempdir(), "recording.wav")
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(audio_settings["channels"])
            wf.setsampwidth(audio_settings["sample_width"])
            wf.setframerate(audio_settings["rate"])
            wf.writeframes(b''.join(recording_frames))
        
        await ctx.send("🎧 Transcribing audio...")
        
        # Transcribe
        transcript = await transcribe_wav(temp_wav)
        
        if not transcript:
            await ctx.send("❌ Could not transcribe any speech from the audio.")
            return
        
        await ctx.send(f"📝 **Heard:** {transcript}")
        
        # Get AI response
        await ctx.send("🤖 Generating response...")
        reply_text = await generate_reply(transcript)
        
        if not reply_text:
            await ctx.send("❌ AI didn't generate a response.")
            return
        
        await ctx.send(f"💭 **AI Says:** {reply_text}")
        
        # Generate and play TTS
        voice_client = ctx.voice_client
        if voice_client and voice_client.is_connected():
            tts_mp3 = os.path.join(tempfile.gettempdir(), "reply.mp3")
            await synthesize_tts(reply_text, tts_mp3)
            
            # Wait if already playing
            while voice_client.is_playing():
                await asyncio.sleep(0.5)
            
            audio_source = discord.FFmpegPCMAudio(tts_mp3)
            voice_client.play(audio_source)
            await ctx.send("🔊 Playing response in voice channel!")
        else:
            await ctx.send("⚠️ Not connected to voice. Use `!join` to hear the response.")
        
        # Cleanup
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
    except Exception as e:
        await ctx.send(f"❌ Error processing audio: {e}")
        logging.error(f"Processing error: {e}", exc_info=True)


@bot.command(name="test")
async def test(ctx: commands.Context) -> None:
    """Test TTS playback in voice channel."""
    if not ctx.voice_client:
        await ctx.send("⚠️ I need to be in a voice channel first! Run `!join`.")
        return

    await ctx.send("🔊 Generating test audio...")
    
    test_phrase = "Testando, testando! Um, dois, três!"
    tts_mp3 = os.path.join(tempfile.gettempdir(), "test_audio.mp3")
    
    try:
        await synthesize_tts(test_phrase, tts_mp3)

        while ctx.voice_client.is_playing():
            await asyncio.sleep(0.5)

        audio_source = discord.FFmpegPCMAudio(tts_mp3)
        ctx.voice_client.play(audio_source)
        
        await ctx.send("✅ Speaking in the voice channel!")
        
    except Exception as e:
        await ctx.send(f"❌ Error during test playback: {e}")


@bot.command(name="leave")
async def leave(ctx: commands.Context) -> None:
    """Stop recording (if active) and leave voice channel."""
    # Stop recording if active
    if recording_thread and recording_thread.is_alive():
        stop_recording_event.set()
        await asyncio.sleep(0.5)
    
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Disconnected from voice.")
    else:
        await ctx.send("⚠️ I'm not in a voice channel.")


@bot.event
async def on_ready() -> None:
    """Bot startup event."""
    logging.info(f"🤖 Bot logged in as {bot.user}")
    logging.info("📋 Available commands: !join, !start, !stop, !test, !leave")


bot.run(DISCORD_TOKEN)
