import requests

#x = requests.get("http://localhost:9000/inference/", data={
#    'a': 1,
#    'b': 2,
#})

#y = requests.get("http://localhost:9000/manager_info")
z = requests.get("http://localhost:9000/inference", data={
    "hello": "world"
    })
x = z.json()


print(type(x))
for k,v in x.items():
    print(f'{k} -> {v}')


#print(x.json())
#print(y.text)
#print(y.json())
