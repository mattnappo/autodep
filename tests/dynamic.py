import numpy as np
import threading
import time
import base64
import json
import sys
import requests
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

with open("./images/cat.png", "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode('utf-8')

CLASSIFICATION_PAYLOAD = {
    "data": {
        "B64Image": {
            "image": image,
        }
    },
    "inference_type": {"ImageClassification":{"top_n":3}}
}

IMG2IMG_PAYLOAD = {
    "data": {
        "B64Image": {
            "image": image,
        }
    },
    "inference_type": "ImageToImage"
}

def secs_nano_to_secs(seconds, nanoseconds):
    return seconds + nanoseconds * 1e-9

def execute_request():
    #logging.info(f"------STARTED NEW THREAD------\nargs:{args}\n")
    headers = {'Content-type': 'application/json'}
    if CLASSIFICATION:
        data = json.dumps(CLASSIFICATION_PAYLOAD)
    else:
        data = json.dumps(IMG2IMG_PAYLOAD)

    res = requests.post("http://localhost:9000/inference", data=data, headers=headers)
    try:
        inference = res.json()

        req_time = res.elapsed.total_seconds()
        inference_time = secs_nano_to_secs(inference[1]['secs'], inference[1]['nanos'])

        overhead_ms = (req_time - inference_time) * 1000
        if PRINT_INFERENCE:
            logging.info(inference)
        else:
            logging.info(overhead_ms)
    except KeyError:
        pass

PRINT_INFERENCE = True
CLASSIFICATION = True

def sine():
    # Sine curve parameters
    duration = 10    # Total duration to run the function
    interval = 0.75  # Time interval to adjust the number of threads
    max_threads = 8  # Maximum number of threads to run at the peak

    # Generate a sine wave over the duration
    times = np.linspace(0, duration, int(duration / interval))
    thread_counts = np.sin((np.pi / duration) * times) * max_threads / 2 + max_threads / 2
    thread_counts = thread_counts.astype(int)  # Convert to int

    plt.plot(times, thread_counts)
    plt.show()
    input("Press enter to start test")

    # Start time
    start_time = time.time()

    # Run the loop for the specified duration
    for count in tqdm(thread_counts):
        threads = []
        for _ in range(count):
            thread = threading.Thread(target=execute_request)
            thread.start()
            threads.append(thread)
        
        # Wait for the interval before adjusting the number of threads
        time.sleep(interval)
        
        # Join threads from the previous interval
        for thread in threads:
            thread.join()

    # End time
    end_time = time.time()

    print(f"Finished executing the function for {end_time - start_time} seconds on a Sine distribution")

def gaussian():
    # Gaussian distribution parameters
    mean = 3  # The peak of the Gaussian curve (at 5 seconds)
    std_dev = 1  # Standard deviation to control the spread
    duration = 8  # Total duration to run the function
    interval = 0.25  # Time interval to adjust the number of threads
    amp = 5

    # Generate a Gaussian distribution over the duration
    times = np.linspace(0, duration, int(duration / interval))
    thread_counts = np.exp(-(times - mean) ** 2 / (2 * std_dev ** 2))
    thread_counts = (thread_counts / thread_counts.max() * amp).astype(int)  # Scale and convert to int

    plt.plot(times, thread_counts)
    plt.show()
    input("Press enter to start test")

    # Start time
    start_time = time.time()

    # Run the loop for the specified duration
    for count in thread_counts:
        threads = []
        for _ in range(count):
            thread = threading.Thread(target=execute_request)
            thread.start()
            threads.append(thread)
        
        # Wait for the interval before adjusting the number of threads
        time.sleep(interval)
        
        # Join threads from the previous interval
        for thread in threads:
            thread.join()

    # End time
    end_time = time.time()

    print(f"Finished executing the function for {end_time - start_time} seconds on a Gaussian curve")

if sys.argv[1] == "gaussian":
    gaussian()
elif sys.argv[1] == "sine":
    sine()
else:
    print("usage:\n\tpython tests.py gaussian|sine\n")

