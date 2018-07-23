import json
json_path1 = "../nielian/artificial/000004.json"
json_path2 = "../nielian/artificial/000001.json"
with open(json_path1, 'r') as f:
    context1 = json.load(f) 
with open(json_path2, 'r') as f:
    context2 = json.load(f) 

# print(type(context1))
for k in context1:
    if context1[k] != context2[k]:
        print(k, context1[k])
        print(k, context2[k])


print(len(context2["FaceControllerLambda"]))





