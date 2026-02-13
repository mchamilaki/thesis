from typing import TypedDict, Optional
from langchain_core.tools import tool

class Invoice(TypedDict):
    account_id: str
    amount_due: float
    due_date: Optional[str]
    status: str  # "paid" | "unpaid" | "overdue" | "not_found"

@tool
def fetch_invoice(account_id: str) -> Invoice:
    """Fetch the latest invoice for an account_id from the billing system (mock)."""
    mock_db = {
        "123": {"account_id": "123", "amount_due": 45.50, "due_date": "2026-02-28", "status": "unpaid"},
        "456": {"account_id": "456", "amount_due": 0.0, "due_date": None, "status": "paid"},
        "789": {"account_id": "789", "amount_due": 120.99, "due_date": "2026-02-20", "status": "overdue"},
    }
    return mock_db.get(account_id, {"account_id": account_id, "amount_due": 0.0, "due_date": None, "status": "not_found"})
