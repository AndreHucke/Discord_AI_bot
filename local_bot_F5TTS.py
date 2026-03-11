import asyncio
import logging
import os
import tempfile
import threading

import discord
import ollama  
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import numpy as np
import scipy.signal
from discord.ext import commands
from dotenv import load_dotenv
import soundfile as sf
from f5_tts.api import F5TTS
from pynput import keyboard

logging.basicConfig(level=logging.INFO)
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b") 

if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN environment variable.")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

# 1. Faster-Whisper Initialization
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")

# 2. F5-TTS Initialization (Custom PT-BR Model)
try:
    logging.info("⏳ Loading Custom PT-BR F5-TTS model on GPU 1...")
    
    # Explicitly path to your downloaded files
    f5tts = F5TTS(
        # ckpt_file="pt_model.safetensors",
        device="cuda" # Keeps this heavy model pinned to your second GPU!
    )
    
    logging.info("✅ F5-TTS custom model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load F5-TTS: {e}")

# Audio Capture Settings
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt32  
LOOPBACK_DEVICE_ID = 13  

# Recording state
recording_frames = []
recording_thread = None
stop_recording_event = threading.Event()
audio_settings = None
active_context = None


async def generate_reply(prompt: str) -> str:
    """Generate AI response using local Qwen model via Ollama."""
    safe_prompt = (
        "Você é uma voz divina que guia um druida que nao lembra nada do passado dele. "
        "Voce nem sempre eh claro e constantemente fala de forma enigimatica, divertida ou caotica. "
        "O texto a seguir é uma transcrição de áudio de uma sessão de RPG ao vivo. "
        "Forneça uma resposta curta e divertida (uma frase) incorporando essa voz. "
        "Nao use emoticons na sua resposta. "
        "Se não houver motivo para responder, diga apenas IGNORE.\n\n"
        f"Os jogadores disseram: {prompt}"
    )
    
    response = await asyncio.to_thread(
        ollama.chat,
        model=QWEN_MODEL_NAME,
        messages=[{'role': 'user', 'content': safe_prompt}]
    )
    
    return response.get('message', {}).get('content', '').strip()


async def transcribe_audio_memory(audio_data: np.ndarray) -> str:
    """Transcribe audio directly from a NumPy array using faster-whisper and VAD."""
    segments, _ = await asyncio.to_thread(
        stt_model.transcribe, 
        audio_data, 
        language="pt",
        vad_filter=True  
    )
    
    text = "".join(segment.text for segment in segments)
    return text.strip()


async def synthesize_tts(text: str, output_wav: str) -> None:
    """Generate TTS using F5-TTS for zero-shot voice cloning."""
    def _generate():
        # The exact text spoken in your reference audio. 
        # If left empty (""), F5-TTS will load a second Whisper model into VRAM to transcribe it.
        # It is highly recommended to hardcode the transcript here to save memory and time.
        reference_transcript = "A transcrição exata do seu áudio de referência deve ser colocada aqui."
        
        wav, sr, _ = f5tts.infer(
            ref_file="divine_voice.wav",
            ref_text=reference_transcript,
            gen_text=text
        )
        sf.write(output_wav, wav, sr)
        
    await asyncio.to_thread(_generate)


def capture_audio_thread():
    """Background thread that captures all audio until stopped."""
    global recording_frames, audio_settings
    
    p = pyaudio.PyAudio()
    
    try:
        device_info = p.get_device_info_by_index(LOOPBACK_DEVICE_ID)
        rate = int(device_info["defaultSampleRate"])
        channels = device_info["maxInputChannels"]
        
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


async def handle_start(ctx):
    """Core logic for starting the recording."""
    global recording_frames, recording_thread, stop_recording_event
    
    if recording_thread and recording_thread.is_alive():
        await ctx.send("⚠️ Already recording! Use `!stop` or `Ctrl+Alt+X` to finish.")
        return
    
    recording_frames = []
    stop_recording_event.clear()
    
    recording_thread = threading.Thread(target=capture_audio_thread, daemon=True)
    recording_thread.start()
    
    await ctx.send("🎙️ Recording started! Use `!stop` or `Ctrl+Alt+X` when you're done.")


async def handle_stop(ctx):
    """Core logic for stopping, processing in-memory, and playing audio."""
    global recording_frames, recording_thread, audio_settings
    
    if not recording_thread or not recording_thread.is_alive():
        await ctx.send("⚠️ No recording in progress.")
        return
    
    stop_recording_event.set()
    await ctx.send("⏹️ Stopping recording and processing...")
    
    await asyncio.to_thread(recording_thread.join, timeout=2.0)
    
    if not recording_frames or not audio_settings:
        await ctx.send("❌ No audio was captured.")
        return
    
    try:
        await ctx.send("🎧 Processing audio in memory...")
        
        raw_data = b''.join(recording_frames)
        audio_np = np.frombuffer(raw_data, dtype=np.int32)
        
        audio_float32 = audio_np.astype(np.float32) / 2147483648.0
        
        if audio_settings["channels"] > 1:
            audio_float32 = audio_float32.reshape(-1, audio_settings["channels"]).mean(axis=1)
            
        original_rate = audio_settings["rate"]
        target_rate = 16000
        if original_rate != target_rate:
            num_samples = int(len(audio_float32) * float(target_rate) / original_rate)
            audio_16k = scipy.signal.resample(audio_float32, num_samples)
        else:
            audio_16k = audio_float32
            
        transcript = await transcribe_audio_memory(audio_16k)
        
        if not transcript:
            await ctx.send("❌ Could not transcribe any speech from the audio.")
            return
        
        await ctx.send(f"📝 **Heard:** {transcript}")
        await ctx.send("🤖 Generating response...")
        
        reply_text = await generate_reply(transcript)
        
        if not reply_text:
            await ctx.send("❌ AI didn't generate a response.")
            return
            
        if "IGNORE" in reply_text.upper():
            await ctx.send("🤫 *The Divine Voice chooses to remain silent.*")
            return
        
        await ctx.send(f"💭 **AI Says:** {reply_text}")
        
        voice_client = ctx.voice_client
        if voice_client and voice_client.is_connected():
            tts_wav = os.path.join(tempfile.gettempdir(), "reply.wav")
            await synthesize_tts(reply_text, tts_wav)
            
            while voice_client.is_playing():
                await asyncio.sleep(0.5)
            
            audio_source = discord.FFmpegPCMAudio(tts_wav)
            voice_client.play(audio_source)
            await ctx.send("🔊 Playing response in voice channel!")
        else:
            await ctx.send("⚠️ Not connected to voice. Use `!join` to hear the response.")
            
    except Exception as e:
        await ctx.send(f"❌ Error processing audio: {e}")
        logging.error(f"Processing error: {e}", exc_info=True)


def on_hotkey_start():
    if active_context and bot.loop:
        bot.loop.call_soon_threadsafe(lambda: bot.loop.create_task(handle_start(active_context)))


def on_hotkey_stop():
    if active_context and bot.loop:
        bot.loop.call_soon_threadsafe(lambda: bot.loop.create_task(handle_stop(active_context)))


@bot.command(name="join")
async def join(ctx: commands.Context) -> None:
    global active_context
    active_context = ctx 

    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("⚠️ Join a voice channel first, then run !join.")
        return

    try:
        if ctx.voice_client and ctx.voice_client.channel != ctx.author.voice.channel:
            await ctx.voice_client.move_to(ctx.author.voice.channel)
        else:
            await ctx.author.voice.channel.connect(timeout=15.0) 
        
        await ctx.send("✅ Connected! You can now use `Ctrl+Alt+S` to start and `Ctrl+Alt+X` to stop recording.")
    except Exception as e:
        await ctx.send(f"❌ Failed to connect: {e}")


@bot.command(name="start")
async def start_cmd(ctx: commands.Context) -> None:
    global active_context
    active_context = ctx
    await handle_start(ctx)


@bot.command(name="stop")
async def stop_cmd(ctx: commands.Context) -> None:
    global active_context
    active_context = ctx
    await handle_stop(ctx)


@bot.command(name="test")
async def test(ctx: commands.Context) -> None:
    if not ctx.voice_client:
        await ctx.send("⚠️ I need to be in a voice channel first! Run `!join`.")
        return

    await ctx.send("🔊 Generating test audio...")
    test_phrase = "Testando, testando! Um, dois, três!"
    tts_wav = os.path.join(tempfile.gettempdir(), "test_audio.wav")
    
    try:
        await synthesize_tts(test_phrase, tts_wav)

        while ctx.voice_client.is_playing():
            await asyncio.sleep(0.5)

        audio_source = discord.FFmpegPCMAudio(tts_wav)
        ctx.voice_client.play(audio_source)
        await ctx.send("✅ Speaking in the voice channel!")
        
    except Exception as e:
        await ctx.send(f"❌ Error during test playback: {e}")


@bot.command(name="leave")
async def leave(ctx: commands.Context) -> None:
    global active_context
    
    if recording_thread and recording_thread.is_alive():
        stop_recording_event.set()
        await asyncio.sleep(0.5)
    
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        active_context = None 
        await ctx.send("👋 Disconnected from voice.")
    else:
        await ctx.send("⚠️ I'm not in a voice channel.")


@bot.event
async def on_ready() -> None:
    logging.info(f"🤖 Bot logged in as {bot.user}")
    logging.info("📋 Available commands: !join, !start, !stop, !test, !leave")
    logging.info("⌨️  Hotkeys active: Ctrl+Alt+S (Start), Ctrl+Alt+X (Stop)")


listener = keyboard.GlobalHotKeys({
    '<ctrl>+<alt>+s': on_hotkey_start,
    '<ctrl>+<alt>+x': on_hotkey_stop
})
listener.start()

bot.run(DISCORD_TOKEN)