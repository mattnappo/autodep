import requests, json
from collections import Counter

while True:
    res = requests.get("http://localhost:9000/workers/status")
    status = res.json()
    count = Counter(status.values())
    working = count['Working']
    if working > 1:
        print("super parallel")
    #print(json.dumps(status.json(), indent=2))
    print()
