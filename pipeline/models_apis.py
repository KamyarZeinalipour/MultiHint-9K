# models_apis.py
# ---------------------------------------------------------------------
# Helper functions to call every LLM used by generating_script.py
# ---------------------------------------------------------------------

from openai import OpenAI
#from mistralai.client import MistralClient
from dotenv import load_dotenv
import google.generativeai as genai
import os


try:
    # New SDK (v1.0+)
    from mistralai import Mistral
except ImportError:
    try:
        # Old SDK (v0.x)
        from mistralai.client import MistralClient as Mistral
    except ImportError:
        # If library is missing entirely, define a dummy class to prevent NameError 
        # (The script will crash only if you try to use Mistral, which is better)
        print("⚠️ Warning: 'mistralai' library not found. Mistral model will fail.")
        Mistral = None

load_dotenv()

load_dotenv()          # pulls API keys & proxy vars from .env

# ---------------------------------------------------------------------
# shared helpers / constants
# ---------------------------------------------------------------------


def render_prompt(template: str, passage: str) -> str:
    """
    Returns the prompt followed by the passage, separated by a blank line.
    """
    return f"{template.rstrip()}\n\nPassage:\n{passage}"


# ---------------------------------------------------------------------
# 1) DeepSeek-Chat  (OpenAI-compatible)
# ---------------------------------------------------------------------
def deepseek_chat(text: str, prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user",   "content": render_prompt(prompt, text)},
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# 2) DeepSeek-Reasoner  (OpenAI-compatible)
# ---------------------------------------------------------------------
def deepseek_reasoner(text: str, prompt: str) -> str:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user",   "content": render_prompt(prompt, text)},
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# 3) GPT-4o  (OpenAI native)
# ---------------------------------------------------------------------
def gpt4o(text: str, prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",           # full-size GPT-4o
        messages=[
          
            {"role": "user",   "content": render_prompt(prompt, text)},
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# 4) GPT-4o-mini-2025-04-16  (OpenAI native, smaller + cheaper)
# ---------------------------------------------------------------------
def gpto4_mini(text: str, prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
    
            {"role": "user",   "content": render_prompt(prompt, text)},
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# 5) Mistral-Large-Latest  (Mistral AI SDK)
# ---------------------------------------------------------------------
def mistral(text: str, prompt: str) -> str:
    if Mistral is None:
        raise ImportError("The 'mistralai' python package is not installed.")
        
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)
    
    # Try new SDK syntax first, fallback to old if method missing
    if hasattr(client, "chat") and hasattr(client.chat, "complete"):
        # New SDK
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": render_prompt(prompt, text)}]
        )
        return response.choices[0].message.content
    else:
        # Old SDK
        response = client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": render_prompt(prompt, text)}]
        )
        return response.choices[0].message.content

def gemini(text: str, prompt: str) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Switched to a valid model name. 
    # Options: "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
    model = genai.GenerativeModel("gemini-2.5-flash") 
    
    response = model.generate_content(
        render_prompt(prompt, text)
    )
    return response.text


