import base64, requests, json, sys

with open(sys.argv[1], "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "data": {
        "B64Image": {
            "image": image,
        }
    },
    "inference_type": "ImageToImage"
}

headers = {"Content-Type": "Application/JSON"}
res = requests.post("http://localhost:9000/inference", data=json.dumps(payload), headers=headers)
inference = res.json()

latency = res.elapsed.total_seconds() - (inference[1]['secs'] + inference[1]['nanos'] * 1e-9)
print(f"latency: {latency * 1000} ms")

with open("output.png", "wb") as f:
    f.write(base64.decodebytes(inference[0]['B64Image']['image'].encode('utf-8')))
print("wrote inference to output.png")
