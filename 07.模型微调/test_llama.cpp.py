import json
import urllib.request

def query_model(prompt, model="qwen3:latest", url="http://192.168.9.179:11434/api/chat"):
    data = {                                                               
        "model": model,
        "option":{
            "seed": 123, # for deterministic responses
            "temperature": 0, # for deterministic responses
        },
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    payload = json.dumps(data).encode("utf-8")                             
    request = urllib.request.Request(url, data=payload, method="POST")     
    request.add_header("Content-Type", "application/json")                 

    response_data = ""
    with urllib.request.urlopen(request) as response:                      
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


result = query_model(prompt = "你好啊！！！");
print(result);
