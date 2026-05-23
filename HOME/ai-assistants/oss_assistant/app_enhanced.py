"""
Enhanced OSS Assistant (Bonus)
Adds:
  - Long-term memory via ChromaDB (vector store)
  - Tool use: web search via DuckDuckGo + calculator
  - Guardrails layer
  - Observability tracing

Run: python app_enhanced.py
Requires: pip install gradio huggingface_hub chromadb duckduckgo-search
"""

import os, json, re, sys
import gradio as gr
from huggingface_hub import InferenceClient

# Add current directory to path for smooth modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def load_dotenv():
    # Search for .env in current, parent, or grandparent folder
    for path in [".env", "../.env", "../../.env"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")
            break

load_dotenv()

from guardrails import safety_check
from observability import traced_call, get_stats

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
MODEL_ID  = "Qwen/Qwen2.5-72B-Instruct"
MAX_TOKENS = 600

# ── Memory (ChromaDB) ──────────────────────────────────────────────────────
try:
    import chromadb
    _chroma = chromadb.Client()
    _mem_col = _chroma.get_or_create_collection("assistant_memory")
    MEMORY_ENABLED = True
except ImportError:
    MEMORY_ENABLED = False

def save_to_memory(user: str, assistant: str, session_id: str = "default"):
    if not MEMORY_ENABLED:
        return
    import hashlib
    uid = hashlib.md5(f"{session_id}{user}".encode()).hexdigest()
    _mem_col.upsert(
        documents=[f"User: {user}\nAssistant: {assistant}"],
        ids=[uid],
        metadatas=[{"session": session_id}],
    )

def recall_memory(query: str, n: int = 3) -> str:
    if not MEMORY_ENABLED or _mem_col.count() == 0:
        return ""
    results = _mem_col.query(query_texts=[query], n_results=min(n, _mem_col.count()))
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else ""

# ── Tools ──────────────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        snippets = [f"- {r['title']}: {r['body'][:200]}" for r in results]
        return "Web search results:\n" + "\n".join(snippets)
    except Exception as e:
        return f"Search unavailable: {e}"

def calculator(expr: str) -> str:
    try:
        # Sanitise: only allow numbers, operators, parens, spaces, dots
        safe = re.sub(r"[^0-9+\-*/().\s]", "", expr)
        return f"Result: {eval(safe)}"   # noqa: S307
    except Exception:
        return "Could not evaluate expression."

TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}

SYSTEM_PROMPT = """You are a helpful personal assistant with access to two tools:
- web_search(query) — search the web for current information
- calculator(expr)  — evaluate a math expression

To use a tool, output EXACTLY this format (nothing else on that line):
TOOL: tool_name(argument)

After receiving the tool result, continue your response normally.
Otherwise, respond helpfully and honestly. Refuse harmful requests."""

# ── Core chat ──────────────────────────────────────────────────────────────
client = InferenceClient(token=HF_TOKEN or None)

def run_tool_if_needed(text: str) -> tuple[str, str | None]:
    """Check if the model wants to call a tool. Returns (result_or_empty, tool_call_str)."""
    m = re.search(r"TOOL:\s*(\w+)\((.+?)\)", text)
    if not m:
        return "", None
    name, arg = m.group(1).strip(), m.group(2).strip().strip("\"'")
    fn = TOOLS.get(name)
    if fn:
        return fn(arg), m.group(0)
    return f"Unknown tool: {name}", m.group(0)


def chat(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not user_message.strip():
        return "", history

    # 1. Guardrail Safety Pre-Classifier Check
    is_safe, safety_reason = safety_check(user_message)
    if not is_safe:
        reply = f"⛔ Refused by Safety Guardrail. ({safety_reason})"
        
        # Log blocked query using observability
        traced_call(
            model="Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": user_message}],
            call_fn=lambda msgs: None,
            extract_reply=lambda r: reply,
            safety_blocked=True,
        )
        
        history = history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return "", history

    # 2. Recall long-term memory
    memory_context = recall_memory(user_message)
    sys_prompt = SYSTEM_PROMPT
    if memory_context:
        sys_prompt += f"\n\nRelevant past context:\n{memory_context}"

    messages = [{"role": "system", "content": sys_prompt}]
    messages += history[-(10 * 2):]
    messages.append({"role": "user", "content": user_message})

    try:
        # Wrap Hugging Face client call with timing/observability
        def make_call(msgs):
            try:
                return client.chat_completion(
                    model=MODEL_ID, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.7,
                )
            except Exception as e:
                if "403" in str(e) or "Forbidden" in str(e) or "authentication" in str(e) or "permission" in str(e):
                    print("Qwen 403 Token Forbidden error detected. Retrying with a public unauthenticated client...")
                    public_client = InferenceClient(token=None)
                    return public_client.chat_completion(
                        model=MODEL_ID, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.7,
                    )
                raise e
            
        def extract(resp):
            return resp.choices[0].message.content.strip()

        reply, meta = traced_call(
            model="Qwen2.5-72B-Instruct",
            messages=messages,
            call_fn=make_call,
            extract_reply=extract,
        )

        # 3. Tool use loop (max 1 tool call)
        tool_result, tool_call = run_tool_if_needed(reply)
        if tool_call:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
            
            def make_call_tool(msgs):
                try:
                    return client.chat_completion(
                        model=MODEL_ID, messages=msgs, max_tokens=MAX_TOKENS,
                    )
                except Exception as e:
                    if "403" in str(e) or "Forbidden" in str(e) or "authentication" in str(e) or "permission" in str(e):
                        print("Qwen 403 Tool Token Forbidden error detected. Retrying with a public unauthenticated client...")
                        public_client = InferenceClient(token=None)
                        return public_client.chat_completion(
                            model=MODEL_ID, messages=msgs, max_tokens=MAX_TOKENS,
                        )
                    raise e
                
            reply, meta2 = traced_call(
                model="Qwen2.5-72B-Instruct",
                messages=messages,
                call_fn=make_call_tool,
                extract_reply=extract,
            )

    except Exception as e:
        reply = f"⚠️ Error: {e}"

    # 4. Save turn to long-term memory
    save_to_memory(user_message, reply)
    
    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return "", history


# ── Live stats function ───────────────────────────────────────────────────
def refresh_stats() -> str:
    try:
        stats = get_stats()
        model_stats = stats.get("Qwen2.5-72B-Instruct", {})
        if not model_stats:
            return "💡 **System Status:** Ready. Start chatting to populate real-time SQL observability metrics!"
        return (
            f"📈 **Total Queries:** {model_stats.get('total_calls', 0)}  |  "
            f"⏱️ **Avg Latency:** {model_stats.get('avg_latency_ms', 0.0)} ms  |  "
            f"🛡️ **Safety Blocks:** {model_stats.get('safety_blocks', 0)}  |  "
            f"❌ **Errors:** {model_stats.get('errors', 0)}"
        )
    except Exception as e:
        return f"Failed to retrieve stats: {e}"

CSS_EMERALD = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.15), transparent 45%),
                radial-gradient(circle at 0% 100%, rgba(4, 120, 87, 0.12), transparent 45%),
                #070a0e !important;
    color: #e2e8f0 !important;
}

footer { display: none !important; }

#chatbot {
    background: rgba(11, 19, 23, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5) !important;
    height: 480px !important;
}

#chatbot .message-wrap .message.user {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border-radius: 18px 18px 2px 18px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    border: none !important;
}

#chatbot .message-wrap .message.assistant {
    background: rgba(22, 32, 45, 0.85) !important;
    color: #f1f5f9 !important;
    border-radius: 18px 18px 18px 2px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.gradio-container input, .gradio-container textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f8fafc !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4) !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
}

.gradio-container button.secondary {
    background: rgba(31, 41, 55, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.gradio-container button.secondary:hover {
    background: rgba(55, 65, 81, 0.8) !important;
    color: white !important;
}

/* Styled Observability Accordion */
.gradio-container .accordion {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(16, 185, 129, 0.15) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35) !important;
    backdrop-filter: blur(12px) !important;
}
"""

# ── UI ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Enhanced OSS Assistant") as demo:
    gr.Markdown("""
    ## 🟢 Enhanced OSS Assistant — Qwen2.5 + Security + Tools + Memory
    A complete production-ready showcase featuring **ChromaDB memory**, **duckduckgo search + calculator**, **input safety guardrails**, and **SQLite observability tracing**.
    """)
    
    with gr.Accordion("📊 Live System Observability Metrics", open=True):
        stats_md = gr.Markdown(refresh_stats())
        
    chatbot  = gr.Chatbot(elem_id="chatbot", show_label=False)
    state    = gr.State([])
    
    with gr.Row():
        msg   = gr.Textbox(placeholder="Ask anything or test a jailbreak…", show_label=False, scale=8, container=False)
        send  = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)
        
    # Wire interactive event chains with reactive stats updates
    send.click(chat, [msg, state], [msg, state]).then(
        lambda h: h, [state], [chatbot]
    ).then(
        refresh_stats, [], [stats_md]
    )
    
    msg.submit(chat, [msg, state], [msg, state]).then(
        lambda h: h, [state], [chatbot]
    ).then(
        refresh_stats, [], [stats_md]
    )
    
    clear.click(lambda: ([], []), [], [chatbot, state]).then(
        refresh_stats, [], [stats_md]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CSS_EMERALD,
    )
