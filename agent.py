import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

with open("knowledge_base.json") as f:
    KB = json.load(f)

KB_TEXT = f"""
AutoStream - AI-powered video editing SaaS for content creators.

PRICING:
- Basic Plan: {KB['pricing']['basic_plan']['price']} | {KB['pricing']['basic_plan']['videos_per_month']} videos/month | {KB['pricing']['basic_plan']['resolution']} | Features: {', '.join(KB['pricing']['basic_plan']['features'])}
- Pro Plan: {KB['pricing']['pro_plan']['price']} | {KB['pricing']['pro_plan']['videos_per_month']} videos/month | {KB['pricing']['pro_plan']['resolution']} | Features: {', '.join(KB['pricing']['pro_plan']['features'])}

POLICIES:
- Refund: {KB['policies']['refund_policy']}
- Support: {KB['policies']['support']}
- Trial: {KB['policies']['trial']}
""".strip()

def mock_lead_capture(name: str, email: str, platform: str):
    print(f"\n[LEAD CAPTURED] {name} | {email} | {platform}\n")
    return f"Lead captured: {name} ({email}) on {platform}"

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    lead_info: dict
    lead_captured: bool
    collecting_lead: bool

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable.")
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0.3, request_timeout=60, max_retries=3)
    return _llm

SYSTEM_PROMPT = f"""You are an AI sales assistant for AutoStream, a SaaS video editing platform.

KNOWLEDGE BASE:
{KB_TEXT}

RULES:
1. Answer product/pricing questions using ONLY the knowledge base above.
2. Be friendly, concise, and helpful.
3. Never make up features or prices not in the knowledge base.
"""

def detect_high_intent(text: str) -> bool:
    patterns = [
        r"\bsign\s*up\b", r"\bsubscribe\b", r"\bbuy\b", r"\bpurchase\b",
        r"\bget started\b", r"\bstart now\b", r"\bregister\b", r"\bonboard\b",
        r"\bi want to (try|get|start|use)\b", r"\bi('d| would) like to (try|get|start|sign)\b",
        r"\bi'm in\b", r"\blet'?s go\b", r"\bsign me up\b", r"\bcount me in\b",
    ]
    return any(re.search(p, text.lower()) for p in patterns)

def extract_email(text: str):
    match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

PLATFORMS = ["youtube", "instagram", "tiktok", "twitter", "facebook", "linkedin", "twitch", "snapchat"]

def extract_platform(text: str):
    for p in PLATFORMS:
        if p in text.lower():
            return p.capitalize()
    return None

def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    lead_info = dict(state.get("lead_info", {}))
    collecting = state.get("collecting_lead", False)
    captured = state.get("lead_captured", False)

    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    if collecting and not captured:
        if "name" not in lead_info:
            lead_info["name"] = last_user_msg.strip()
            return {
                "messages": [AIMessage(content=f"Thanks, {lead_info['name']}! What's your email address?")],
                "lead_info": lead_info,
                "collecting_lead": True,
                "lead_captured": False,
            }

        elif "email" not in lead_info:
            email = extract_email(last_user_msg)
            if not email:
                return {
                    "messages": [AIMessage(content="I didn't catch a valid email. Could you share it again?")],
                    "lead_info": lead_info,
                    "collecting_lead": True,
                    "lead_captured": False,
                }
            lead_info["email"] = email
            return {
                "messages": [AIMessage(content="Great! Which creator platform do you primarily use? (e.g. YouTube, Instagram, TikTok)")],
                "lead_info": lead_info,
                "collecting_lead": True,
                "lead_captured": False,
            }

        elif "platform" not in lead_info:
            lead_info["platform"] = extract_platform(last_user_msg) or last_user_msg.strip()
            mock_lead_capture(lead_info["name"], lead_info["email"], lead_info["platform"])
            reply = (
                f"You're all set! We've registered your interest in AutoStream Pro.\n"
                f"Our team will reach out to {lead_info['email']} shortly. "
                f"Welcome aboard, {lead_info['name']}!"
            )
            return {
                "messages": [AIMessage(content=reply)],
                "lead_info": lead_info,
                "collecting_lead": False,
                "lead_captured": True,
            }

    if detect_high_intent(last_user_msg) and not captured:
        return {
            "messages": [AIMessage(content="Great choice! To get you started, may I have your full name?")],
            "lead_info": {},
            "collecting_lead": True,
            "lead_captured": False,
        }

    chat_history = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    ai_response = get_llm().invoke(chat_history)
    return {
        "messages": [AIMessage(content=ai_response.content)],
        "lead_info": lead_info,
        "collecting_lead": collecting,
        "lead_captured": captured,
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()

def main():
    print("=" * 55)
    print("  AutoStream AI Sales Assistant (type 'quit' to exit)")
    print("=" * 55)

    app = build_graph()
    state: AgentState = {
        "messages": [],
        "lead_info": {},
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
        )
        print(f"\nAutoStream Agent: {last_ai}")

if __name__ == "__main__":
    main()
