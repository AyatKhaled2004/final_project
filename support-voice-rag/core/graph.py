"""
The multi-agent flow, built with LangGraph.

  question -> rewrite -> router -+-> knowledge agent (PDF search + grounded answer)
                                 +-> chitchat agent  (greetings, thanks)
                                 +-> handoff agent   (human needed -> ticket)
"""
from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import config
from core.store import format_source


class State(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]   # [(role, text), ...]
    session_id: str
    standalone: str
    route: str
    answer: str
    sources: list[str]
    handoff: bool


REWRITE_PROMPT = (
    "Rewrite the customer's last message as one standalone question, using the conversation "
    "for context (resolve words like 'it', 'that', 'this one'). Keep the same language. "
    "Return only the question."
)

ROUTER_PROMPT = (
    "Classify the customer's message into exactly one label:\n"
    "knowledge - a question about products, orders, shipping, returns, payments, accounts, warranty, policies\n"
    "chitchat - greeting, thanks, goodbye, small talk\n"
    "human - the customer asks for a human/agent, is angry, threatens legal action, reports fraud, "
    "or needs something only a person can do\n"
    "Answer with the label only."
)

KNOWLEDGE_PROMPT = (
    "You are a polite customer-service assistant. Answer ONLY from the context below.\n"
    "Reply in the same language as the customer's message. Keep it short and clear, in plain "
    "sentences that sound natural when read aloud. No markdown, no lists.\n"
    "If the context does not contain the answer, say you are not sure and offer to connect "
    "the customer with a human agent. Never invent policies, prices or deadlines.\n\n"
    "Context:\n{context}"
)

CHITCHAT_PROMPT = (
    "You are a friendly customer-service assistant. Reply briefly in the customer's language. "
    "Do not state any facts about products or policies. Offer help with their order or question."
)

HANDOFF_PROMPT = (
    "Apologize briefly and tell the customer you are passing the conversation to a human agent "
    "who will follow up. Reply in the customer's language, in one or two short sentences."
)

NO_CONTEXT_ANSWER = (
    "I could not find that in our documents. Would you like me to connect you with a human agent?"
)


def _as_messages(history: list[tuple[str, str]]):
    out = []
    for role, text in history[-config.HISTORY_TURNS * 2:]:
        out.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return out


def parse_route(text: str) -> str:
    t = text.strip().lower()
    if "human" in t:
        return "human"
    if "chitchat" in t or "small" in t:
        return "chitchat"
    return "knowledge"  # safest default: answer from documents


def build_graph(llm, kb, on_handoff: Callable[[dict], None] | None = None):
    def rewrite(state: State) -> State:
        history = state.get("history", [])
        if not history:
            return {"standalone": state["question"]}
        convo = "\n".join(f"{r}: {t}" for r, t in history[-config.HISTORY_TURNS * 2:])
        msg = llm.invoke(
            [SystemMessage(REWRITE_PROMPT), HumanMessage(f"{convo}\nuser: {state['question']}")]
        )
        return {"standalone": msg.content.strip() or state["question"]}

    def router(state: State) -> State:
        msg = llm.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(state["standalone"])])
        return {"route": parse_route(msg.content)}

    def knowledge(state: State) -> State:
        docs = kb.search(state["standalone"], config.TOP_K)
        if not docs:
            return {"answer": NO_CONTEXT_ANSWER, "sources": [], "handoff": False}
        context = "\n\n".join(f"[{format_source(d)}]\n{d.page_content}" for d in docs)
        msg = llm.invoke(
            [SystemMessage(KNOWLEDGE_PROMPT.format(context=context))]
            + _as_messages(state.get("history", []))
            + [HumanMessage(state["question"])]
        )
        sources = list(dict.fromkeys(format_source(d) for d in docs))
        return {"answer": msg.content.strip(), "sources": sources, "handoff": False}

    def chitchat(state: State) -> State:
        msg = llm.invoke(
            [SystemMessage(CHITCHAT_PROMPT)]
            + _as_messages(state.get("history", []))
            + [HumanMessage(state["question"])]
        )
        return {"answer": msg.content.strip(), "sources": [], "handoff": False}

    def handoff(state: State) -> State:
        msg = llm.invoke([SystemMessage(HANDOFF_PROMPT), HumanMessage(state["question"])])
        if on_handoff:
            on_handoff(
                {
                    "session_id": state.get("session_id", ""),
                    "question": state["question"],
                    "conversation": state.get("history", [])[-config.HISTORY_TURNS * 2:],
                }
            )
        return {"answer": msg.content.strip(), "sources": [], "handoff": True}

    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("router", router)
    graph.add_node("knowledge", knowledge)
    graph.add_node("chitchat", chitchat)
    graph.add_node("handoff", handoff)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"knowledge": "knowledge", "chitchat": "chitchat", "human": "handoff"},
    )
    for node in ("knowledge", "chitchat", "handoff"):
        graph.add_edge(node, END)
    return graph.compile()
