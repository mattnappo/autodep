import base64
import requests
import json
import logging
import threading
import time

with open("./images/seg1.jpg", "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode('utf-8')

#print(image)

payload = {
    "data": {
        "B64Image": {
            "image": image,
        }
    },
    "inference_type": "ImageToImage"
}

def secs_nano_to_secs(seconds, nanoseconds):
    return seconds + nanoseconds * 1e-9

def thread_function(args):
    print(f"------STARTED NEW THREAD------\nargs:{args}\n")
    headers = {'Content-type': 'application/json'}
    data = json.dumps(payload)

    times = []
    for i in range(NUM_REQS_PER_THREAD):
        res = requests.post("http://localhost:9000/inference", data=data, headers=headers)
        inference = res.json()

        req_time = res.elapsed.total_seconds()
        inference_time = secs_nano_to_secs(inference[1]['secs'], inference[1]['nanos'])

        overhead_ms = (req_time - inference_time) * 1000
        print(overhead_ms)
        times.append(overhead_ms)
    return times

NUM_THREADS = 6
NUM_REQS_PER_THREAD = 10

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.info("Main")

    threads = []
    for i in range(NUM_THREADS):
        x = threading.Thread(target=thread_function, args=(i,))
        threads.append(x)
        x.start()

    times = []
    for i, x in enumerate(threads):
        t = x.join()
        times.append(t)
        logging.info("Thread %s: finished", i)

    print("done")
    print(times)

