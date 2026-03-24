from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import settings
from agent.guardrails import check_guardrails


def create_agent(tools: list = None):
    tools = tools or []
    llm = ChatOpenAI(
        model=settings.model_name,
        base_url=settings.vllm_url,
        api_key="not-needed",
    ).bind_tools(tools)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda s: {"messages": [llm.invoke(s["messages"])]})
    graph.add_node("guardrails", check_guardrails)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "guardrails", END: END},
    )
    graph.add_edge("guardrails", "tools")
    graph.add_edge("tools", "agent")

    # MemorySaver enables interrupt()/resume for the confirmation gate.
    # Swap for SqliteSaver later if you want persistence across restarts.
    return graph.compile(checkpointer=MemorySaver())
