import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": "Hi, who are you?",
        "stream": False  
    }
)

data = response.json()
print(data["response"])
