# AutoStream – Social-to-Lead Agentic Workflow

An AI-powered conversational sales agent for **AutoStream** (a fictional SaaS video editing platform) that identifies high-intent users, answers product questions via RAG, and captures leads using a tool call.

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd social-to-lead-agentic-workflow
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your Gemini API key**

On Windows:
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
```

On macOS/Linux:
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

**4. Run the agent**
```bash
python agent.py
```

**5. Example conversation**
```
You: Hi, what plans do you offer?
You: What's included in the Pro plan?
You: That sounds great, I want to try the Pro plan for my YouTube channel.
You: John Doe
You: john@example.com
You: YouTube
```

---

## Architecture Explanation

**Why LangGraph?**
LangGraph was chosen over AutoGen because it provides explicit, inspectable state management via a typed `AgentState` dictionary. Each conversation turn flows through a single `agent` node that reads and writes state deterministically — making it easy to track multi-step lead collection (name → email → platform) without losing context. LangGraph's graph-based design also makes it straightforward to add new nodes (e.g., a CRM integration node) later.

**How State is Managed**
The `AgentState` TypedDict holds four fields across all turns:
- `messages` — full conversation history (persisted via LangGraph's `add_messages` reducer, giving the LLM memory across 5–6 turns)
- `lead_info` — dict accumulating collected fields (name, email, platform)
- `collecting_lead` — boolean flag that switches the agent into lead-collection mode
- `lead_captured` — boolean flag that prevents duplicate tool calls

**RAG Pipeline**
The knowledge base (`knowledge_base.json`) is loaded at startup and injected directly into the LLM's system prompt as structured text. This is a lightweight RAG approach — no vector DB needed for a small, static knowledge base. For a larger KB, this would be replaced with a FAISS/Chroma retriever.

**Intent Detection**
A keyword-based classifier (`detect_high_intent`) screens each user message for purchase signals. When triggered, the agent shifts into lead-collection mode and collects fields one at a time before calling `mock_lead_capture()`.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

1. **WhatsApp Business API (Meta Cloud API)** — Register a WhatsApp Business account and get a phone number + access token from [Meta for Developers](https://developers.facebook.com/).

2. **Webhook endpoint** — Build a small FastAPI/Flask server with two routes:
   - `GET /webhook` — for Meta's verification handshake (returns the `hub.challenge` token)
   - `POST /webhook` — receives incoming messages as JSON payloads

3. **Message routing** — On each `POST`, extract the sender's `wa_id` (phone number) and message text. Use `wa_id` as a session key to load/save that user's `AgentState` (stored in Redis or a simple dict for prototyping).

4. **Invoke the agent** — Pass the message into `app.invoke(state)`, get the AI reply, and send it back via a `POST` to `https://graph.facebook.com/v18.0/<phone-number-id>/messages` with the recipient's `wa_id`.

5. **State persistence** — Since WhatsApp conversations are async and stateless per request, serialize `AgentState` to Redis (keyed by `wa_id`) after every turn and deserialize it at the start of the next turn to maintain full conversation memory.

```
WhatsApp User
     │  sends message
     ▼
Meta Webhook POST /webhook
     │
     ▼
FastAPI Server
  ├─ Load state from Redis (keyed by wa_id)
  ├─ app.invoke(state)  ← LangGraph agent
  ├─ Save updated state to Redis
  └─ POST reply → Meta Graph API → WhatsApp User
```
