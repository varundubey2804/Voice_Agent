import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from backend.llm.service import LLMService
from backend.rag.retrieval import RAGService
from backend.core.logger import logger

# Define the state for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str

class AssistantAgent:
    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def retrieve_node(self, state: AgentState):
        """Retrieves context from RAG based on the latest user message."""
        latest_message = state["messages"][-1].content
        try:
            docs = self.rag_service.search(latest_message, top_k=3)
            context = "\n".join(docs)
        except Exception as e:
            logger.error(f"Error in retrieve_node: {e}")
            context = ""

        return {"context": context}

    def generate_node(self, state: AgentState):
        """Generates a response using the LLM and retrieved context."""
        latest_message = state["messages"][-1].content
        context = state.get("context", "")

        system_prompt = (
            "You are a helpful, privacy-preserving local AI assistant. "
            "Use the following context to answer the user's question if relevant.\n"
            f"Context:\n{context}\n"
        )

        try:
            # Note: The LLMService uses string prompts natively right now.
            # In a full LangChain setup, we'd pass the actual BaseMessage objects to a ChatModel.
            # For simplicity with the custom LLMService wrapper:
            response_text = self.llm_service.generate_sync(prompt=latest_message, system_prompt=system_prompt)
            return {"messages": [AIMessage(content=response_text)]}
        except Exception as e:
            logger.error(f"Error in generate_node: {e}")
            return {"messages": [AIMessage(content="I am having trouble processing that right now.")]}

    async def invoke(self, message: str) -> str:
        """Invokes the agent graph asynchronously."""
        state = {"messages": [HumanMessage(content=message)], "context": ""}
        try:
            # Assuming invoke is synchronous in this minimal langgraph version, we wrap it
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.graph.invoke, state)

            # Get the last message which should be the AI response
            final_message = result["messages"][-1].content
            return final_message
        except Exception as e:
            logger.error(f"Error invoking agent graph: {e}")
            return "An error occurred while reasoning."
