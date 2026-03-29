import json
import requests
import asyncio
import os
from pathlib import Path
from email import message_from_file
from client import EmailTriageEnv
from server.models import EmailAction

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM = """You are an expert email triage assistant. Read the email and respond ONLY with raw JSON, no explanation or markdown fences.

== ACTION SCHEMAS ==
For classify: 
{"action_type":"classify","email_id":"<id>","urgency":"urgent|normal|low|spam","priority":1-5}

For triage:
{"action_type":"triage","email_id":"<id>","urgency":"urgent|normal|low|spam","priority":1-5,"reply_draft":"<50+ word reply>","route_to":"support|sales|engineering|hr|finance|general"}

== ROUTING GUIDE ==
engineering — server/API outages, security bugs
finance     — invoices, legal data requests (GDPR)
support     — customer complaints, refund requests
sales       — pipeline reviews, new partnership proposals
hr          — onboarding, benefits, team culture
general     — broad news, lunch menu, newsletters
"""

def parse_eml(file_path):
    """Simple parser to extract subject, sender, and body from an .eml file."""
    with open(file_path, "r", errors="ignore") as f:
        msg = message_from_file(f)
    
    subject = msg.get("Subject", "No Subject")
    sender = msg.get("From", "Unknown Sender")
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body = part.get_payload(decode=True).decode(errors="ignore")
                break
        if not body:  # fallback to html if no plain text
             for part in msg.walk():
                if part.get_content_type() == "text/html":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")
    
    return {
        "id": Path(file_path).stem,
        "subject": subject,
        "sender": sender,
        "body": body[:2000] # truncate very long emails for the LLM
    }

def ask_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model": "mistral",
        "prompt": SYSTEM + "\n\n" + prompt,
        "stream": False
    })
    raw = r.json()["response"].strip()
    
    # Improved robust JSON extraction
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in response: {raw}")
    
    # Use JSONDecoder.raw_decode to extract the first valid JSON object
    try:
        data, index = json.JSONDecoder().raw_decode(raw[start:])
        return EmailAction(**data)
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to older method if raw_decode fails for some reason
        end = raw.rfind("}")
        if end != -1:
            try:
                data = json.loads(raw[start : end + 1])
                return EmailAction(**data)
            except:
                pass
        raise ValueError(f"Failed to parse JSON: {e}\nRaw response: {raw}")

# ── Paste YOUR real emails here ──────────────────────────────────────────────
MY_EMAILS = [
    {
        "id": "my_001",
        "subject": "Urgent: Server Down",
        "sender":  "admin@company.com",
        "body":    "The main production server is not responding to any requests. Please investigate immediately.",
    },
]
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Detect any .eml files in the current folder
    eml_files = [f for f in os.listdir(".") if f.endswith(".eml")]
    processed_emails = [parse_eml(f) for f in eml_files]
    
    # Combine with hardcoded ones
    all_emails_to_test = MY_EMAILS + processed_emails

    if not all_emails_to_test:
        print("No emails found to test! Add to MY_EMAILS or drop .eml files here.")
        return

    with EmailTriageEnv("http://localhost:8000") as env:
        for email_data in all_emails_to_test:
            # inject your email into an easy task
            result = env.reset(task_id="easy")
            
            # override the email the server gave with yours
            prompt = f"""Task: triage this email.
email_id: {email_data['id']}
from: {email_data['sender']}
subject: {email_data['subject']}
body: {email_data['body']}"""

            print(f"\n--- Testing: {email_data['subject'][:60]} ---")
            
            try:
                # Run full triage immediately
                action = ask_ollama(prompt)
                
                print(f"Mistral detected: urgency={action.urgency.value if action.urgency else 'none'}, priority={action.priority}")
                print(f"Route to: {action.route_to.value if action.route_to else 'none'}")
                if action.reply_draft:
                    preview = action.reply_draft[:150].replace("\n", " ")
                    print(f"Reply Preview: {preview}...")
            except Exception as e:
                print(f"Error processing {email_data['id']}: {e}")

if __name__ == "__main__":
    main()
