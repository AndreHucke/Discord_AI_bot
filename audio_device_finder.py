import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
default_idx = wasapi_info["defaultOutputDevice"]

print(f"--- YOUR DEFAULT OUTPUT DEVICE IS ID: {default_idx} ---")

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev["isLoopbackDevice"] or dev["maxInputChannels"] > 0:
        print(f"ID {i}: {dev['name']} (Channels: {dev['maxInputChannels']})")

p.terminate()