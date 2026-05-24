from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)