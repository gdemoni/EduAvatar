"""Define a custom workflow based on the provided flowchart."""

from typing import Any, Dict, Literal
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from react_agent.context import Context
from react_agent.prompts import (
    DIRECT_SOLVE_PROMPT,
    INTENT_PROMPT,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_CHECK_PROMPT,
    REWRITE_PROMPT,
    SIMPLE_SOLVE_CHECK_PROMPT,
    SOLVE_PROMPT,
    STEPS_PROMPT,
)
from react_agent.state import InputState, State
from react_agent.tools import faiss_search_local, search
from react_agent.utils import load_chat_model


# 1. 意图识别节点
async def intent_recognition_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze user intent: 'explain', 'solve', 'welcome' or 'ask_intent'."""
    # Check if there are no messages (initial state) or the last message is empty
    if not state.messages:
        return {"intent": "welcome"}
    
    last_message = state.messages[-1]
    if isinstance(last_message.content, list):
        # Handle cases where content is a list of blocks (e.g. text + image)
        text_content = "".join([block.get("text", "") for block in last_message.content if block.get("type") == "text"])
        if not text_content.strip():
             return {"intent": "welcome"}
    elif isinstance(last_message.content, str):
        if not last_message.content.strip():
            return {"intent": "welcome"}

    # Get model name from config or use default
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)

    model = load_chat_model(model_name)
    # Using the last user message for intent classification
    response = await model.ainvoke([SystemMessage(content=INTENT_PROMPT)] + state.messages)
    intent = response.content.strip().lower()
    
    if "ask_intent" in intent:
        return {"intent": "ask_intent"}
    if "welcome" in intent:
        return {"intent": "welcome"}
    if "explain" in intent or "讲解" in intent:
        return {"intent": "explain"}
    return {"intent": "solve"}



# 2.5 询问用户节点 (New Node for Clarification)
async def ask_clarification_node(state: State) -> Dict[str, Any]:
    """Ask user for clarification on intent."""
    if state.intent == "ask_intent":
        message = "请问您是希望我为您讲解某个知识点，还是解答具体的题目？"
    else:
        message = "请明确您的需求。"
        
    return {"messages": [AIMessage(content=message)]}


async def welcome_node(state: State) -> Dict[str, Any]:
    """Send a welcome message."""
    message = "你好！我是你的智能助教。我可以为你讲解知识点，例如解释什么是光合作用；也可以为你解答具体的题目，请直接把题目发给我。请告诉我你想学什么？"
    return {"messages": [AIMessage(content=message)]}


# 3. 题目解答节点 (Branch 2) - Initial Analysis
async def problem_solving_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Decide whether a solve query should be answered directly or via retrieval."""
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)

    model = load_chat_model(model_name)
    force_local = os.environ.get("FORCE_LOCAL_RETRIEVAL", "").strip().lower() in {"1", "true", "yes"}
    if force_local:
        return {"solve_mode": "retrieval"}
    human_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    query = " ".join(str(c) for c in human_messages) if human_messages else state.messages[-1].content

    check_response = await model.ainvoke([
        SystemMessage(content=SIMPLE_SOLVE_CHECK_PROMPT.format(query=query))
    ])
    decision = str(check_response.content).strip().upper()

    if "SIMPLE" in decision:
        return {"solve_mode": "direct", "data_source": "direct"}
    return {"solve_mode": "retrieval"}


# 4. 知识检索节点 (Split into Local and Web)
async def retrieve_local_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Retrieve knowledge from Local FAISS."""
    # Get model name from config or use default
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)
         
    model = load_chat_model(model_name)
    
    force_local = os.environ.get("FORCE_LOCAL_RETRIEVAL", "").strip().lower() in {"1", "true", "yes"}
    # 1. Rewrite Query
    human_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    history = "\n".join(str(c) for c in human_messages) if human_messages else state.messages[-1].content
    
    rewrite_response = await model.ainvoke([
        SystemMessage(content=QUERY_REWRITE_PROMPT.format(history=history))
    ])
    query = rewrite_response.content.strip() or history

    # 2. Search Local
    results = await faiss_search_local(query)
    
    context = ""
    source = "none"
    
    if results and "results" in results and results["results"]:
        raw_context = "\n\n".join([r.get("content", "") for r in results["results"]])
        context = raw_context
        source = "local"
    
    return {"retrieved_context": context, "data_source": source}


async def retrieve_web_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Retrieve knowledge from Web Search if local search fails."""
    # Get model name from config or use default
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)
         
    model = load_chat_model(model_name)
    
    # 1. Rewrite Query (Reuse history logic or just use last query)
    # Ideally we reuse the rewritten query but state doesn't store it yet. 
    # Let's re-generate for simplicity or use the same logic.
    human_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    history = "\n".join(str(c) for c in human_messages) if human_messages else state.messages[-1].content
    
    rewrite_response = await model.ainvoke([
        SystemMessage(content=QUERY_REWRITE_PROMPT.format(history=history))
    ])
    query = rewrite_response.content.strip()

    # 2. Search Web
    web_results = await search(query)
    context = ""
    source = "none"
    
    if web_results and "results" in web_results:
         context = "\n\n".join([r.get("content", "") for r in web_results["results"]])
         source = "web"
    else:
         context = ""
         source = "none"
    
    return {"retrieved_context": context, "data_source": source}


# 5. 分层改写节点 (Branch 1)
async def layered_rewriting_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rewrite content."""
    # Get model name from config or use default
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)
         
    model = load_chat_model(model_name)
    
    source_desc = "本地知识库" if state.data_source == "local" else "网络搜索"
    if state.data_source == "none":
        source_desc = "无参考资料"

    prompt = REWRITE_PROMPT.format(context=state.retrieved_context, source=source_desc)
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"rewritten_content": response.content}


# 6. 生成解题步骤节点 (Branch 2)
async def generate_solution_steps_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate detailed solution steps."""
    # Get model name from config or use default
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)
         
    model = load_chat_model(model_name)
    # Use all human messages to form the query
    human_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    query = " ".join(str(c) for c in human_messages) if human_messages else state.messages[-1].content
    
    # Step 1: Generate full solution
    solve_prompt = SOLVE_PROMPT.format(query=query, context=state.retrieved_context)
    solution_response = await model.ainvoke([SystemMessage(content=solve_prompt)])
    solution = solution_response.content
    
    # Step 2: Extract/Format steps
    source_desc = "本地知识库" if state.data_source == "local" else "网络搜索"
    if state.data_source == "none":
        source_desc = "无参考资料"

    steps_prompt = STEPS_PROMPT.format(solution=solution, source=source_desc)
    steps_response = await model.ainvoke([SystemMessage(content=steps_prompt)])
    
    return {"solution_steps": steps_response.content}


async def _direct_solve_node(state: State, config: Dict[str, Any] = None) -> Dict[str, Any]:
    model_name = "openai/qwen-plus"
    if config and "configurable" in config:
        model_name = config["configurable"].get("model", model_name)
    elif config and "model" in config:
         model_name = config.get("model", model_name)

    model = load_chat_model(model_name)
    human_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)]
    query = " ".join(str(c) for c in human_messages) if human_messages else state.messages[-1].content
    response = await model.ainvoke([SystemMessage(content=DIRECT_SOLVE_PROMPT.format(query=query))])
    return {"solution_steps": response.content, "data_source": "direct"}


# 7. 输出教学内容给学生
async def output_teaching_content_node(state: State) -> Dict[str, Any]:
    """Format the final output."""
    if state.intent == "explain":
        content = state.rewritten_content
    else:
        content = state.solution_steps

    source_sentence = "以上内容主要参考了本地知识库中的资料。"
    if state.data_source == "direct":
        source_sentence = "这个问题比较基础，我是直接推理计算后给出的答案。"
    elif state.data_source == "web":
        source_sentence = "以上内容主要参考了网络搜索的结果。"
    elif state.data_source == "none":
        source_sentence = "抱歉，我没有找到相关的资料。"

    if source_sentence not in content:
        content = f"{content.rstrip()} {source_sentence}".strip()

    return {"messages": [AIMessage(content=content)]}


# Conditional Edge Logic
def _route_intent(state: State) -> Literal["retrieve_local_explain", "problem_solving_node", "ask_clarification_node", "welcome_node"]:
    if state.intent == "ask_intent":
        return "ask_clarification_node"
    if state.intent == "welcome":
        return "welcome_node"
    if state.intent == "explain":
        return "retrieve_local_explain"
    return "problem_solving_node"

def _check_retrieval_explain(state: State) -> Literal["retrieve_web_node", "layered_rewriting_node"]:
    if state.data_source == "local" and state.retrieved_context.strip():
        return "layered_rewriting_node"
    return "retrieve_web_node"

def _check_retrieval_solve(state: State) -> Literal["retrieve_web_node", "generate_solution_steps_node"]:
    if state.data_source == "local" and state.retrieved_context.strip():
        return "generate_solution_steps_node"
    return "retrieve_web_node"

def _route_solve_mode(state: State) -> Literal["direct_solve_node", "retrieve_local_solve"]:
    if state.solve_mode == "direct":
        return "direct_solve_node"
    return "retrieve_local_solve"

def _route_after_web(state: State) -> Literal["layered_rewriting_node", "generate_solution_steps_node"]:
    if state.intent == "explain":
        return "layered_rewriting_node"
    return "generate_solution_steps_node"


# Build the Graph
builder = StateGraph(State, input_schema=InputState, config_schema=Context)

# Add Nodes
builder.add_node("intent_recognition_node", intent_recognition_node)
builder.add_node("ask_clarification_node", ask_clarification_node)
builder.add_node("welcome_node", welcome_node) # New Node
builder.add_node("problem_solving_node", problem_solving_node)

# We use the same function for retrieval but register it as two separate nodes 
# to maintain the distinct branches in the visual graph/logic
builder.add_node("retrieve_local_explain", retrieve_local_node)
builder.add_node("retrieve_local_solve", retrieve_local_node)
builder.add_node("retrieve_web_node", retrieve_web_node)


builder.add_node("layered_rewriting_node", layered_rewriting_node)
builder.add_node("generate_solution_steps_node", generate_solution_steps_node)
builder.add_node("direct_solve_node", _direct_solve_node)
builder.add_node("output_teaching_content_node", output_teaching_content_node)

# Add Edges
builder.add_edge("__start__", "intent_recognition_node")
builder.add_conditional_edges("intent_recognition_node", _route_intent)

# Branch 1: Explain
builder.add_conditional_edges("retrieve_local_explain", _check_retrieval_explain)
builder.add_edge("layered_rewriting_node", "output_teaching_content_node")

# Branch 2: Solve
builder.add_conditional_edges("problem_solving_node", _route_solve_mode)
builder.add_conditional_edges("retrieve_local_solve", _check_retrieval_solve)
builder.add_edge("generate_solution_steps_node", "output_teaching_content_node")
builder.add_edge("direct_solve_node", "output_teaching_content_node")

# Web Search Edges
builder.add_conditional_edges("retrieve_web_node", _route_after_web)

# Clarification End
builder.add_edge("ask_clarification_node", END)
builder.add_edge("welcome_node", END)

# End
builder.add_edge("output_teaching_content_node", END)

graph = builder.compile(name="RAG Agent")
