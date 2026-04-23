from dotenv import load_dotenv
load_dotenv()

import agent
from langchain_core.messages import HumanMessage, AIMessage

app = agent.build_graph()
state = {"messages": [], "lead_info": {}, "lead_captured": False, "collecting_lead": False}

turns = [
    "Hi, what plans do you offer?",
    "What is included in the Pro plan?",
    "I want to sign up for the Pro plan",
    "John Doe",
    "john@example.com",
    "YouTube",
]

for msg in turns:
    print(f"You: {msg}")
    state["messages"] = state["messages"] + [HumanMessage(content=msg)]
    state = app.invoke(state)
    last_ai = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "")
    print(f"Agent: {last_ai}")
    print()
