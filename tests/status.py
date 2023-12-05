import requests, json, time
from collections import Counter

while True:
    res = requests.get("http://localhost:9000/workers")
    status = res.json()
    print(status)
    print()
    time.sleep(0.05)
