"""
Email dataset for the Customer Support Agent Environment.
Each entry has: id, body, category, keywords, tone, urgency, expected_resolution.
Diversity across billing, complaints, queries — varied phrasing and urgency levels.
"""

EMAIL_DATA = [
    # ── BILLING ────────────────────────────────────────────────────────────────
    {
        "id": "email_001",
        "email": "I was charged twice for the same order #ORD-4821. My bank statement shows two identical debits of $49.99 on March 3rd. Please fix this immediately.",
        "category": "billing",
        "keywords": ["refund", "charged", "duplicate", "payment"],
        "tone": "professional",
        "urgency": "high",
        "expected_resolution": "Acknowledge double charge, confirm refund initiation within 3-5 business days."
    },
    {
        "id": "email_002",
        "email": "Hi, I cancelled my subscription last month but I'm still being billed $15/month. This is the third time I'm writing. Please stop charging me.",
        "category": "billing",
        "keywords": ["cancel", "subscription", "refund", "charge"],
        "tone": "frustrated",
        "urgency": "high",
        "expected_resolution": "Confirm subscription cancellation, issue refund for incorrect charges."
    },
    {
        "id": "email_003",
        "email": "Your website says the product costs $29 but I was charged $39. I have a screenshot of the advertised price. This feels like a bait and switch.",
        "category": "billing",
        "keywords": ["price", "charged", "advertised", "refund"],
        "tone": "accusatory",
        "urgency": "medium",
        "expected_resolution": "Apologize for price discrepancy, honor advertised price, refund the difference."
    },

    # ── COMPLAINT ───────────────────────────────────────────────────────────────
    {
        "id": "email_004",
        "email": "My order #ORD-9034 was supposed to arrive 5 days ago. The tracking page just says 'in transit' with no updates. I need this for an event tomorrow.",
        "category": "complaint",
        "keywords": ["delay", "tracking", "order", "urgent", "arrived"],
        "tone": "urgent",
        "urgency": "critical",
        "expected_resolution": "Apologize for delay, investigate shipment, provide updated ETA or replacement."
    },
    {
        "id": "email_005",
        "email": "I received a completely wrong item. I ordered a blue medium t-shirt and received a red XL jacket. The packaging was correct but contents were wrong. Very disappointed.",
        "category": "complaint",
        "keywords": ["wrong item", "replacement", "sorry", "disappointed"],
        "tone": "disappointed",
        "urgency": "medium",
        "expected_resolution": "Apologize, arrange return, ship correct item with expedited delivery."
    },
    {
        "id": "email_006",
        "email": "The product I bought stopped working after just two weeks. This is clearly a manufacturing defect. I want a replacement or full refund.",
        "category": "complaint",
        "keywords": ["defect", "replacement", "refund", "warranty", "broken"],
        "tone": "assertive",
        "urgency": "medium",
        "expected_resolution": "Acknowledge defect, offer replacement or full refund under warranty."
    },

    # ── QUERY ───────────────────────────────────────────────────────────────────
    {
        "id": "email_007",
        "email": "Hi, I forgot my account password and the reset email isn't arriving. I've checked my spam folder. Can you help me regain access?",
        "category": "query",
        "keywords": ["reset", "password", "account", "access", "help"],
        "tone": "polite",
        "urgency": "medium",
        "expected_resolution": "Guide through alternative verification, manually trigger reset or verify email address."
    },
    {
        "id": "email_008",
        "email": "Do you ship internationally? I'm in India and want to order your premium plan. What are the shipping costs and expected delivery times to Mumbai?",
        "category": "query",
        "keywords": ["international", "shipping", "delivery", "India"],
        "tone": "curious",
        "urgency": "low",
        "expected_resolution": "Confirm international shipping availability, provide costs and estimated delivery time."
    },
    {
        "id": "email_009",
        "email": "Can I upgrade from the Basic plan to Pro mid-month? Will I be charged the full amount or prorated? And can I keep all my current data?",
        "category": "query",
        "keywords": ["upgrade", "plan", "prorated", "data", "billing"],
        "tone": "informational",
        "urgency": "low",
        "expected_resolution": "Confirm prorated billing on upgrades, assure data retention during plan change."
    },
    {
        "id": "email_010",
        "email": "I'm trying to integrate your API with my app but getting a 401 Unauthorized error even though my API key looks correct. Is there a known issue?",
        "category": "query",
        "keywords": ["API", "error", "key", "authentication", "integration"],
        "tone": "technical",
        "urgency": "high",
        "expected_resolution": "Troubleshoot 401 error: check key scopes, regenerate if needed, confirm endpoint."
    },
]