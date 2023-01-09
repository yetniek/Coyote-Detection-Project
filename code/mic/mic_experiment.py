# importing all the modules
import pyaudio
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
import utils
import numpy as np
import time
import datetime
import serial

FORMAT = pyaudio.paInt16 # 16 bits per sample
CHANNELS = 1 # Number of audio channels
RATE = 44100 # Record at 44100 samples per second
CHUNK = 1024 # Record in chunks of 1024 samples
RECORD_SECONDS = 5.1 # Note duration of 5 seconds
WAVE_OUTPUT_FILENAME = "./file.wav"

# port setting
ser = serial.Serial('/dev/ttyUSB_DEV1', 115200, timeout=1)
ser.reset_output_buffer()

while True:
    audio = pyaudio.PyAudio() # Create an Interface to PortAudio
    # Open a new data stream and start recording 
    stream = audio.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=2,
                        frames_per_buffer=CHUNK)

    frames = [] # Initialize array to store frames
    max_sound=0
    time_stamp=0
    print("=========================================")
    print("Start Time: ", datetime.datetime.fromtimestamp(time.time()))

    # Store data in chunks for 5 seconds
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK) # Read data
        frames.append(data) # Add Data
        new_data=np.fromstring(data, dtype=np.int16)
        new_data=int(np.average(np.abs(new_data)))
        if new_data>max_sound:
            max_sound=new_data
            time_stamp=time.time()
    peak_time = datetime.datetime.fromtimestamp(time_stamp) #Save peak times of coyote howling 
    print("Peak Time: ", peak_time)
    print("End Time: ", datetime.datetime.fromtimestamp(time.time()))
    print("=========================================")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    audio.terminate() 

    # Open and save the recorded data as a WAV file 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    # Write and close the file
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    result = utils.predict(WAVE_OUTPUT_FILENAME) # Model predict from WAV output file 
    # parameter : result, peak_time -> transmit
    if result==1: # Coyote Dtected
        print("Coyote Detected")
        ##### Networking Part #####
        # peak_time = peak_time.strftime('%Y-%m-%d %H:%M:%S.%f')[14:26]
        # peak_time = peak_time.encode('utf-8')
        # ser.write(peak_time)
        # time.sleep(1)
        ###########################
    elif result==0: # no detection
        print("Non-Coyote")