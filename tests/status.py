import requests, json, time
from collections import Counter

curr = -1
while True:
    res = requests.get("http://localhost:9000/workers/_status")
    status = res.json()
    count = Counter(status.values())
    working = count['Working']
    if working != curr:
        curr = working
        print(f"Active workers: {curr}")

    time.sleep(0.05)


