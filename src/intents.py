from enum import Enum

class Intent(str, Enum):
    BILLING = "billing"
    USAGE = "usage"
    NETWORK_ISSUE = "network_issue"
    OTHER = "other"
