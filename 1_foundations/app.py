from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# Load environment variables
load_dotenv(override=True)

# -------------------- PUSHOVER FUNCTIONS --------------------
def push(text):
    """Send notifications using Pushover API"""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Record user details"""
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Record unknown questions"""
    push(f"Recording {question}")
    return {"recorded": "ok"}


# -------------------- TOOLS DEFINITIONS --------------------
record_user_details_json = {
    "name": "record_user_details",
    "description": "Record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if provided"},
            "notes": {"type": "string", "description": "Additional context or notes"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unknown question"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# -------------------- MAIN BOT CLASS --------------------
class Me:

    def __init__(self):
        # Load Gemini API key
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")

        # Initialize OpenAI client with Gemini endpoint
        self.gemini = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        self.name = "Mahad Habib Rana"

        # Load LinkedIn profile from PDF
        self.linkedin = ""
        reader = PdfReader("me/profile.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Load career summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)

            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        """System instructions for Gemini"""
        system_prompt = f"""You are acting as {self.name}. 
You are answering questions on {self.name}'s website, particularly about career, background, skills, and experience. 
You must represent {self.name} professionally and faithfully.

If you don't know an answer, use the `record_unknown_question` tool to log it.
If the user seems interested, ask for their email and log it with `record_user_details`.

Here is background context:

## Summary:
{self.summary}

## LinkedIn Profile:
{self.linkedin}

Stay professional and engaging, as if talking to a potential client or employer.
"""
        return system_prompt

    def chat(self, message, history):
        messages = [
            {"role": "system", "content": self.system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]

        done = False
        while not done:
            response = self.gemini.chat.completions.create(
                model="gemini-1.5-flash", 
                messages=messages, 
                tools=tools
            )

            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                message = choice.message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True

        return choice.message.content


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
