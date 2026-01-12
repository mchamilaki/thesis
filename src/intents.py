from enum import Enum

class Intent(str, Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    INFO_LOOKUP = "info_lookup"
    ESCALATION = "escalation"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"
