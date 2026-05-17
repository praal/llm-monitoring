import os
from time import sleep
import requests
import json


def _get_openrouter_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it before running, e.g. export OPENROUTER_API_KEY='sk-or-v1-...'"
        )
    return key

def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    return file.read()

def prompt_model(prompt, model_name, system_prompt = "", temperature = 0.2, max_tokens=20000):
    response = ""
    complete = False
    while not complete:
        try:
          response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
            "Authorization": f"Bearer {_get_openrouter_api_key()}",
            },
            data=json.dumps({
              "model": model_name,  # Optional
              "messages": [
              {
                "role": "user",
                "content": prompt
              },
              {"role": "system",
               "content": system_prompt
              },
              ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }))
          complete = True
        except:
            print("pipeline broke, trying again...", model_name)
            sleep(5)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        print("ERRRRRR", response.json())
        return ""
