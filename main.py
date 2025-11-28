# main.py

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool


# Load environment variables (OPENAI_API_KEY from .env or system env)
load_dotenv()


# ---------- Pydantic response model ----------

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]


# ---------- LLM + parser ----------

# Use any OpenAI model you have access to
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or "gpt-4.1-mini", etc.
    temperature=0,
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# ---------- Prompt for the agent ----------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that helps generate a concise research-style summary.

            You have access to the following tools:
            - A simulated web search (search_tool)
            - A simulated Wikipedia lookup (wiki_tool)
            - A save tool (save_tool) to write content to disk

            When useful, call one or more tools to gather information.
            However, avoid infinite loops: call each tool at most a couple of times.

            After you finish reasoning and tool use, you MUST output ONLY a JSON object
            that matches this schema:

            {format_instructions}

            Do not include any extra commentary, markdown, or text outside the JSON.
            """,
        ),
        # Space for chat history if you later want to add it
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ---------- Tools + agent ----------

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    # Safety: avoid endless tool loops
    max_iterations=10,
)


# ---------- Main entry point ----------

def main():
    user_query = input("What can I help you research? ")

    # Run the agent
    raw_response = agent_executor.invoke({"query": user_query})

    # Expected shape: {"output": "<final JSON text>", "intermediate_steps": [...]}
    output_text = raw_response.get("output", "")

    # Guard: handle "agent stopped" or other non-JSON outputs
    if isinstance(output_text, str) and "Agent stopped due to max iterations" in output_text:
        print(
            "The agent hit the maximum number of iterations without finishing.\n"
            "Try re-running with a simpler question or improving the tools."
        )
        print("\nRaw output:\n", output_text)
        return

    try:
        structured_response = parser.parse(output_text)

        print("\nStructured ResearchResponse:\n")
        print(structured_response.model_dump_json(indent=2))

    # --- SAVE TO TXT FILE ---
        save_path = "research_output.txt"
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(structured_response.model_dump_json(indent=2))
            f.write("\n\n" + "="*80 + "\n\n")

        print(f"\nSaved to {save_path}\n")

    except Exception as e:
        print("Error parsing model output into ResearchResponse:", e)
        print("\nRaw output text was:\n", repr(output_text))
        print("\nFull raw response dict:\n", raw_response)


if __name__ == "__main__":
    main()
