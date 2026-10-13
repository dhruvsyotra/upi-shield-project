from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    END = "__END__"
    StateGraph = None


class AgentState(TypedDict, total=False):
    receiver_upi: str
    risk_score: float
    anomaly_score: float
    report_count: int
    verified: bool
    decision: str
    lien_status: str
    action_log: list[str]


def evaluate_risk(state: AgentState):
    risk = float(state.get("risk_score", 0))
    anomaly = float(state.get("anomaly_score", 0))
    reports = int(state.get("report_count", 0))
    verified = bool(state.get("verified", False))
    if risk >= 75 or (anomaly >= 0.80 and reports >= 2 and not verified):
        decision = "HOLD"
    elif risk >= 45 or anomaly >= 0.55 or reports >= 2:
        decision = "REVIEW"
    else:
        decision = "SAFE"
    return {"decision": decision, "action_log": state.get("action_log", []) + [f"Risk evaluated: {decision}"]}


def route(state: AgentState):
    return state["decision"]


def apply_action(state: AgentState):
    d = state["decision"]
    if d == "HOLD":
        lien = "LIEN_ACTIVE"
        message = f"Downstream transfer hold simulated for {state['receiver_upi']} pending review."
    elif d == "REVIEW":
        lien = "REVIEW_QUEUE"
        message = f"Downstream outflow monitoring intensified for {state['receiver_upi']}."
    else:
        lien = "NO_LIEN"
        message = "No guardrail intervention required."
    return {"lien_status": lien, "action_log": state.get("action_log", []) + [message]}


class LocalGuardrailApp:
    def invoke(self, state: AgentState):
        state = {**state, **evaluate_risk(state)}
        state = {**state, **apply_action(state)}
        return state


def build_guardrail():
    if not LANGGRAPH_AVAILABLE:
        return LocalGuardrailApp()
    graph = StateGraph(AgentState)
    graph.add_node("evaluate", evaluate_risk)
    graph.add_node("hold", apply_action)
    graph.add_node("review", apply_action)
    graph.add_node("safe", apply_action)
    graph.set_entry_point("evaluate")
    graph.add_conditional_edges("evaluate", route, {"HOLD": "hold", "REVIEW": "review", "SAFE": "safe"})
    graph.add_edge("hold", END)
    graph.add_edge("review", END)
    graph.add_edge("safe", END)
    return graph.compile()

GUARDRAIL_APP = build_guardrail()
