import requests

#x = requests.get("http://localhost:9000/inference/", data={
#    'a': 1,
#    'b': 2,
#})

#y = requests.get("http://localhost:9000/manager_info")
y = requests.get("http://localhost:9000/inference")
#z = requests.get("http://localhost:9000/inference", data={
#    "hello": "world"
#    })
#x = z.json()


#print(type(x))
#print(x)
#for k,v in x.items():
#    print(f'{k} -> {v}')


#print(x.json())
#print(y.text)
print("text", y.text)
print(y.json())
