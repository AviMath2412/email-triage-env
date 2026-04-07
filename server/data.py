"""
server/data.py — Shared constants, templates, and task definitions.
"""

from .models import UrgencyLabel, Department

# ---------------------------------------------------------------------------
# Synthetic email dataset with ground-truth labels
# Each tuple: (subject, sender, body, urgency, priority, department)
# ---------------------------------------------------------------------------

EMAIL_TEMPLATES = [
    (
        "URGENT: Production database down — customers cannot login",
        "oncall@ops.company.com",
        "We are seeing 100% error rate on login attempts. DB connection pool exhausted. "
        "Affecting ~50k active users. Started 14 minutes ago.",
        UrgencyLabel.URGENT, 5, Department.ENGINEERING,
    ),
    (
        "Invoice #4521 overdue — payment required within 48 hours",
        "billing@vendor.com",
        "Your invoice #4521 for $12,400 was due on March 1st and remains unpaid. "
        "Please arrange payment within 48 hours to avoid service suspension.",
        UrgencyLabel.URGENT, 4, Department.FINANCE,
    ),
    (
        "Re: Job application — Senior Engineer position",
        "candidate@gmail.com",
        "Thank you for the interview last week. I wanted to follow up and reiterate "
        "my strong interest in the Senior Engineer role.",
        UrgencyLabel.NORMAL, 3, Department.HR,
    ),
    (
        "Q1 sales pipeline review — please review before Friday",
        "vp.sales@company.com",
        "Attached is the Q1 pipeline review deck. Please review before our Friday standup. "
        "Key deals: Acme ($200k), Globex ($85k).",
        UrgencyLabel.NORMAL, 3, Department.SALES,
    ),
    (
        "Weekly engineering newsletter — issue #47",
        "newsletter@techdigest.io",
        "This week: Rust surpasses Go in adoption surveys, new LLM benchmarks released, "
        "Kubernetes 1.30 features breakdown.",
        UrgencyLabel.LOW, 1, Department.GENERAL,
    ),
    (
        "Customer complaint: charged twice for subscription",
        "angry.customer@gmail.com",
        "I was charged twice this month for my Pro subscription ($49 × 2 = $98). "
        "This is completely unacceptable. I want an immediate refund. "
        "If not resolved today I'm disputing both charges.",
        UrgencyLabel.URGENT, 5, Department.SUPPORT,
    ),
    (
        "Lunch menu for office this week",
        "office.manager@company.com",
        "This week's lunch options: Monday - Italian, Tuesday - Thai, Wednesday - Mexican. "
        "Please submit orders by 10am each day.",
        UrgencyLabel.LOW, 1, Department.GENERAL,
    ),
    (
        "Security alert: unusual login from new location",
        "security@company.com",
        "We detected a login to your account from Lagos, Nigeria (IP: 41.203.x.x) "
        "at 3:42 AM EST. If this was not you, please secure your account immediately.",
        UrgencyLabel.URGENT, 5, Department.ENGINEERING,
    ),
    (
        "Partnership proposal: co-marketing opportunity",
        "partnerships@startupxyz.com",
        "I'm reaching out about a potential co-marketing partnership. "
        "We have 200k newsletter subscribers in your target market. "
        "Would love to hop on a call to explore synergies.",
        UrgencyLabel.NORMAL, 2, Department.SALES,
    ),
    (
        "Your Amazon order has shipped",
        "shipment-tracking@amazon.com",
        "Your order #112-3456789 has been shipped and will arrive by Thursday. "
        "Tracking number: 1Z999AA10123456784.",
        UrgencyLabel.SPAM, 1, Department.GENERAL,
    ),
    (
        "Performance review deadline this Friday",
        "hr@company.com",
        "Reminder: All managers must submit Q1 performance reviews by EOD Friday. "
        "Please log in to the HR portal to complete your direct reports' evaluations.",
        UrgencyLabel.NORMAL, 3, Department.HR,
    ),
    (
        "Enterprise contract renewal — Globex Corp $180k/year",
        "procurement@globexcorp.com",
        "Globex is prepared to renew for 3 years at $180,000/year if we can get the "
        "new API features on the roadmap. Decision needed by end of month.",
        UrgencyLabel.URGENT, 5, Department.SALES,
    ),
    (
        "Office printer out of toner",
        "facilities@company.com",
        "The 3rd floor printer is out of toner. I've submitted a request to facilities. "
        "Expected replacement tomorrow morning.",
        UrgencyLabel.LOW, 1, Department.GENERAL,
    ),
    (
        "Re: Onboarding — welcome to the team!",
        "hr@company.com",
        "Welcome aboard! Your first day is Monday. Please review the attached onboarding "
        "checklist and come ready with your laptop and government ID.",
        UrgencyLabel.NORMAL, 2, Department.HR,
    ),
    (
        "CRITICAL: API rate limit exceeded — all integrations failing",
        "alerts@monitoring.company.com",
        "Alert triggered: External API rate limit exceeded at 23:47 UTC. "
        "All third-party integrations returning 429 errors. "
        "Revenue impact estimated at $3k/hour. Immediate action required.",
        UrgencyLabel.URGENT, 5, Department.ENGINEERING,
    ),
    (
        "GDPR Data Access Request — Action Required by Law",
        "privacy@legal-firm.com",
        "Our client (ID #8821) has submitted a formal Subject Access Request (SAR) under GDPR. "
        "You are legally required to provide all stored personal data within 30 days. "
        "Failure to comply can result in significant fines.",
        UrgencyLabel.URGENT, 4, Department.FINANCE,
    ),
    (
        "Automated: Out of Office until Monday",
        "john.doe@partner.com",
        "Thank you for your email. I am currently out of the office and will return on Monday. "
        "For urgent matters, please contact support@partner.com.",
        UrgencyLabel.SPAM, 1, Department.GENERAL,
    ),
    (
        "Feedback on your recent purchase #99218",
        "reviews@marketplace.com",
        "We hope you are enjoying your recent purchase! Would you mind taking 30 seconds "
        "to leave a review? Your feedback helps smaller sellers grow.",
        UrgencyLabel.SPAM, 1, Department.GENERAL,
    ),
    (
        "Upcoming team building event: Vote on activities!",
        "culture-committee@company.com",
        "Hi everyone! We're planning our next team off-site. Please vote on your preferred "
        "activities (Bowling, Laser Tag, or Hiking) by Friday at 5pm.",
        UrgencyLabel.LOW, 2, Department.HR,
    ),
    (
        "Urgent issue with employee benefits enrollment",
        "benefits@external-hr.com",
        "We noticed that 12 employees in your department have failed to complete their "
        "mandatory benefits enrollment. The window closes at midnight tonight. "
        "Please follow up with your team immediately.",
        UrgencyLabel.URGENT, 5, Department.HR,
    ),
]

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Single Email Classification",
        "description": (
            "You are an executive assistant. Classify the given email by urgency "
            "(urgent/normal/low/spam) and assign a priority score (1=lowest, 5=highest). "
            "Use action_type='classify' with fields: email_id, urgency, priority."
        ),
        "difficulty": "easy",
        "max_steps": 5,
        "pass_threshold": 0.70,
        "excellent_threshold": 0.90,
    },
    "medium": {
        "id": "medium",
        "name": "Inbox Priority Ranking",
        "description": (
            "You are managing a busy inbox. Sort ALL emails from most to least urgent. "
            "Use action_type='rank' with ranked_ids as a list of all email IDs "
            "ordered from highest priority to lowest. You may refine your ranking across "
            "multiple steps — each submission is scored and you get feedback."
        ),
        "difficulty": "medium",
        "max_steps": 8,
        "pass_threshold": 0.60,
        "excellent_threshold": 0.85,
    },
    "hard": {
        "id": "hard",
        "name": "Full Triage Pipeline",
        "description": (
            "You are an executive assistant. For the given email you must: "
            "(1) classify urgency and priority, "
            "(2) draft a professional reply, "
            "(3) route it to the correct department. "
            "Use action_type='triage' with all fields: email_id, urgency, priority, "
            "reply_draft, route_to."
        ),
        "difficulty": "hard",
        "max_steps": 10,
        "pass_threshold": 0.55,
        "excellent_threshold": 0.80,
    },
}
