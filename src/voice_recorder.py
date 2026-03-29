import threading
import logging
import wave
import pyaudio

logging.basicConfig(level=logging.INFO)

# Configuration hardcoded to prevent Discord bot initialization
LOOPBACK_DEVICE_ID = 11
FORMAT = pyaudio.paInt32
CHUNK_SIZE = 1024
WAVE_OUTPUT_FILENAME = "divine_voice.wav"

stop_recording_event = threading.Event()
recording_frames = []
audio_settings = {}

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
        
        logging.info("🎤 Recording started... Press Enter in the console to stop.")
        
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

def main():
    # Initialize and start the background capture thread
    record_thread = threading.Thread(target=capture_audio_thread)
    record_thread.start()
    
    
    # Wait for the user to press Enter to trigger the stop event
    input()
    stop_recording_event.set()
    record_thread.join()
    
    # Compile frames and save the .wav file
    if recording_frames and audio_settings:
        logging.info(f"💾 Saving audio data to {WAVE_OUTPUT_FILENAME}...")
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(audio_settings["channels"])
            wf.setsampwidth(audio_settings["sample_width"])
            wf.setframerate(audio_settings["rate"])
            wf.writeframes(b''.join(recording_frames))
        logging.info("✅ WAV file saved successfully!")
    else:
        logging.warning("⚠️ No audio data was captured.")

if __name__ == "__main__":
    main()
