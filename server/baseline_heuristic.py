"""
baseline_heuristic.py — Simple keyword-rule agent used for baseline scoring.
"""

from .models import EmailAction, EmailObservation, ActionType, UrgencyLabel

_URGENT_KW = [
    "urgent", "down", "outage", "critical", "immediately", "emergency",
    "overdue", "complaint", "alert", "security", "crisis", "failed"
]
_SPAM_KW = [
    "unsubscribe", "newsletter", "amazon", "shipped", "tracking", "promo", "marketing"
]

def heuristic_agent(obs: EmailObservation, task_id: str) -> EmailAction:
    """
    Simple keyword-rule agent. Used for baseline scoring without an LLM.
    """
    
    def classify_email(email):
        text = (email.subject + " " + email.body).lower()
        if any(k in text for k in _SPAM_KW):
            return UrgencyLabel.SPAM, 1
        if any(k in text for k in _URGENT_KW):
            return UrgencyLabel.URGENT, 5
        return UrgencyLabel.NORMAL, 3

    if task_id == "easy" and obs.single_email:
        e = obs.single_email
        urgency, priority = classify_email(e)
        return EmailAction(
            action_type=ActionType.CLASSIFY,
            email_id=e.id,
            urgency=urgency,
            priority=priority
        )

    elif task_id == "medium":
        def sort_key(e):
            text = (e.subject + " " + e.body).lower()
            if any(k in text for k in _SPAM_KW):
                return 0
            return sum(1 for k in _URGENT_KW if k in text)
        
        ranked = sorted(obs.inbox, key=sort_key, reverse=True)
        return EmailAction(
            action_type=ActionType.RANK,
            ranked_ids=[e.id for e in ranked]
        )

    elif task_id == "hard" and obs.single_email:
        e = obs.single_email
        text = (e.subject + " " + e.body).lower()
        urgency, priority = classify_email(e)
        
        name = e.sender.split("@")[0].replace(".", " ").title()
        reply = (
            f"Dear {name},\n\n"
            f"Thank you for reaching out. We have received your message regarding "
            f'"{e.subject}" and are treating this as a priority matter.\n\n'
            f"Our team is investigating and will respond with a resolution within 4 hours. "
            f"We apologize for any inconvenience caused.\n\n"
            f"Best regards,\nSupport Team"
        )
        
        # Route logic
        if any(k in text for k in ["bug", "server", "database", "code", "api", "security", "login"]):
            route = "engineering"
        elif any(k in text for k in ["invoice", "payment", "billing", "refund", "charge"]):
            route = "finance"
        elif any(k in text for k in ["complaint", "issue", "problem", "help", "charged"]):
            route = "support"
        elif any(k in text for k in ["sales", "contract", "renewal", "pricing", "demo"]):
            route = "sales"
        else:
            route = "general"
            
        return EmailAction(
            action_type=ActionType.TRIAGE,
            email_id=e.id,
            urgency=urgency,
            priority=priority,
            reply_draft=reply,
            route_to=route
        )
    
    return EmailAction(action_type=ActionType.DONE)
