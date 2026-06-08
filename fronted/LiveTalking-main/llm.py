import time
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load backend environment variables
backend_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/agent/.env"))
load_dotenv(backend_env_path)

# Add backend to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/agent/src"))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from basereal import BaseReal
from logger import logger

# Import LangGraph agent
try:
    from react_agent.graph import graph
    from langchain_core.messages import HumanMessage
except ImportError as e:
    logger.error(f"Failed to import react_agent: {e}")
    graph = None


def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()
    
    if not graph:
        logger.error("Graph not loaded, cannot process message.")
        nerfreal.put_msg_txt("系统错误：无法加载智能体。")
        return

    async def run_agent():
        try:
            inputs = {"messages": [HumanMessage(content=message)]}
            # Run the graph
            # Ensure config is provided with a proper Context instance if the graph requires runtime configuration
            # However, standard StateGraph.compile() usually creates a Runnable that accepts config dict.
            # If your graph nodes use runtime.context, we need to ensure the config passed has the 'configurable' key
            # that matches what the graph expects, OR the graph needs to be invoked with a config that allows default context initialization.
            
            # Based on the error 'NoneType' object has no attribute 'model', it seems runtime.context is None.
            # This happens when the graph is compiled with a config_schema (Context) but no config is provided during invoke.
            
            # Let's provide a default configuration
            config = {"configurable": {"model": "openai/qwen-plus"}} 
            
            result = await graph.ainvoke(inputs, config=config)
            
            # Extract the last message content
            if "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                content = last_msg.content
                
                # Stream the content to nerfreal in chunks
                buffer = ""
                for char in content:
                    buffer += char
                    # Only split by sentence terminators to reduce TTS calls and improve fluency
                    if char in ".!?;。！？；\n":
                        if len(buffer) > 10: 
                            logger.info(buffer)
                            nerfreal.put_msg_txt(buffer)
                            buffer = ""
                
                # Send remaining text
                if buffer:
                    logger.info(buffer)
                    nerfreal.put_msg_txt(buffer)
                        
            else:
                logger.warning("No messages returned from agent.")
                
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            nerfreal.put_msg_txt("抱歉，我遇到了一些问题。")

    try:
        # Run the async agent in the current thread (which is an executor thread)
        asyncio.run(run_agent())
    except Exception as e:
        logger.error(f"Asyncio run failed: {e}")

    end = time.perf_counter()
    logger.info(f"llm processing time: {end - start}s")

