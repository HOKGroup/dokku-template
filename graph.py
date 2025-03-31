import logging
from typing import Annotated, Union

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from pydantic import BaseModel, Field
from langgraph.types import Command

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
ASSISTANT_SYSTEM_PROMPT_BASE = """you are helpful marketing assistant working at a architecture firm tasked with extracting information from project interview transcripts"""

weak_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", tags=["assistant"])
model = weak_model
assistant_model = weak_model


class GraphProcessingState(BaseModel):
    # user_input: str = Field(default_factory=str, description="The original user input")
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    prompt: str = Field(
        default_factory=str, description="The prompt to be used for the model"
    )
    transcript: str = Field(default="", description="Uploaded text file content")
    tools_enabled: dict = Field(
        default_factory=dict, description="The tools enabled for the assistant"
    )

    marketing_copy: str = Field(
        default="", description="The result of summarize_transcript tool call"
    )
    metrics: Union[str, dict] = Field(
        default="", description="The result of extract_parameters tool call"
    )
    idml_file: str = Field(default="")


@tool
async def generate_marketing_copy(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[GraphProcessingState, InjectedState],
) -> Command:
    """creates a marketing copy based on user specifications"""

    transcript = state.transcript

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", transcript),
            (
                "system",
                """Generate marketing copy that is compelling and aligned with the provided guidelines.
                focus on key benefits, unique selling points, and engaging narrative Just give the
                marketing copy avoid adding any additional text or explanation the output must be plain text avoid markdown annotations""",
            ),
            ("human", query),
        ]
    )
    chain = prompt | assistant_model
    response = await chain.ainvoke({"messages": state.messages})

    # Extract the content from the AIMessage
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )

    return Command(
        update={
            "marketing_copy": response_content,
            "messages": [
                ToolMessage(content=response_content, tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def extract_metrics(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[GraphProcessingState, InjectedState],
) -> Command:
    """Extract metrics from transcription such as project  project_name, size, height, number_of_floors, completion_date, client_name, project_team_members, external_consultants etc."""

    class Metrics(BaseModel):
        project_name: str = Field(description="Name of the project", default="")
        project_location: str = Field(
            description="Project address or location, the detailed location the better",
            default="",
        )
        size: str = Field(description="Size of the project", default="")
        height: str = Field(description="Height of the project", default="")
        number_of_floors: str = Field(
            description="Number of floors in the project", default=""
        )
        completion_date: str = Field(
            description="Date of project completion", default=""
        )
        client_name: str = Field(description="Name of the client", default="")
        project_team_members: list[str] = []
        external_consultants: list[str] = []

    transcript = state.transcript
    metrics_prompt = """Extract project metrics and statistics from the following transcription.
Focus on these aspects: metrics should only and only be the in proper format avoid adding any description or other things if there is nothing can be found do not put in the output
for consultants and and project team only and only output a result if there is a first name and last or a entity name can be found. avoid general name such as structural consultants, lighting consultant etc.  outputs should be a list of strings for the names 
 only use these keys when possible and relevant location, project_name, size, height, number_of_floors, completion_date, client_name, project_team_members, external_consultants **Transcript**
             """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", transcript),
            ("system", metrics_prompt),
            ("human", query),
        ]
    )
    parser = JsonOutputParser(pydantic_object=Metrics)

    chain = prompt | assistant_model | parser
    response = await chain.ainvoke({"messages": state.messages})

    # Extract the content from the AIMessage
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )

    return Command(
        update={
            "metrics": response_content,
            "messages": [
                ToolMessage(content=response_content, tool_call_id=tool_call_id)
            ],
        }
    )


tools = [
    generate_marketing_copy,
    # extract_metrics,
]


async def assistant_node(state: GraphProcessingState, config=None):
    assistant_tools = []

    if state.tools_enabled.get("generate_marketing_copy", True):
        assistant_tools.append(generate_marketing_copy)
    # if state.tools_enabled.get("extract_metrics", True):
    #     assistant_tools.append(extract_metrics)

    assistant_model = model.bind_tools(assistant_tools)
    if state.prompt:
        final_prompt = "\n".join([state.prompt, ASSISTANT_SYSTEM_PROMPT_BASE])
    else:
        final_prompt = ASSISTANT_SYSTEM_PROMPT_BASE

    # Add transcript context if available
    if state.transcript:
        transcript_context = f"The following is a transcript that's been uploaded by the user:\n\n{state.transcript}\n\n"
        final_prompt = transcript_context + final_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", final_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | assistant_model
    response = await chain.ainvoke({"messages": state.messages}, config=config)

    result = {"messages": response}

    return result


def assistant_cond_edge(state: GraphProcessingState):
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"Tool call detected: {last_message.tool_calls}")
        return "tools"
    return END


async def get_metrics(transcript):
    parser = JsonOutputParser()

    metrics_prompt_template = PromptTemplate(
        template="""Extract project metrics and statistics from the following transcription.
Focus on these aspects: metrics should only and only be the in proper format avoid adding any description or other things if there is nothing can be found do not put in the output
for consultants and and project team only and only output a result if there is a first name and last or a entity name can be found. avoid general name such as structural consultants, lighting consultant etc.  outputs should be a list of strings for the names 
 only use these keys when possible and relevant location, project_name, size, height, number_of_floors, completion_date, client_name, project_team_members, external_consultants



Transcription:
{transcription_text}


Generate a JSON object containing the extracted metrics and statistics.
Be specific and quantitative where possible.""",
        input_variables=["transcription_text"],
    )
    metrics_chain = metrics_prompt_template | model | parser
    metrics_result = await metrics_chain.ainvoke(
        {
            "transcription_text": transcript,
        }
    )

    return metrics_result


async def initial_state_processor(state: GraphProcessingState, config=None):
    """Process the initial state and update it with any necessary initializations"""
    # Initialize tools_enabled if not present

    # Add transcript context if available
    # if state.prompt:
    #     final_prompt = "\n".join([state.prompt, ASSISTANT_SYSTEM_PROMPT_BASE])
    # else:
    #     final_prompt = ASSISTANT_SYSTEM_PROMPT_BASE

    # # Add transcript context if available
    # if state.transcript:
    #     transcript_context = f"The following is a transcript that's been uploaded by the user:\n\n{state.transcript}\n\n"
    #     final_prompt = transcript_context + final_prompt

    if not state.tools_enabled:
        state.tools_enabled = {"generate_marketing_copy": True, "extract_metrics": True}

    # Add any initial system message if needed
    if not state.messages:
        state.messages = []

    # Only extract metrics if we have a transcript
    if state.transcript:
        metrics = await get_metrics(state.transcript)
        state.metrics = metrics
        return {"metrics": metrics}

    return {}


def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph"""
    # Initialize the graph
    workflow = StateGraph(GraphProcessingState)

    # Add nodes
    workflow.add_node("initial_state_processor", initial_state_processor)
    workflow.add_node("assistant_node", assistant_node)
    workflow.add_node("tools", ToolNode(tools))

    # Edges
    workflow.add_edge("initial_state_processor", "assistant_node")
    workflow.add_edge("tools", "assistant_node")

    # Conditional routing
    workflow.add_conditional_edges(
        "assistant_node",
        # If the latest message (result) from assistant is a tool call -> assistant_cond_edge routes to tools
        # If the latest message (result) from assistant is a not a tool call -> assistant_cond_edge routes to END
        assistant_cond_edge,
    )
    # Set entry point to the initial state processor
    workflow.set_entry_point("initial_state_processor")

    return workflow.compile()


graph = define_workflow()
