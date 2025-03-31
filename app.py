#!/usr/bin/env python

import logging
import logging.config
from typing import Any
from uuid import uuid4, UUID
import json
from langchain.globals import set_verbose, set_debug
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig
from pydantic import BaseModel
import docx2txt
import os
from datetime import datetime
import re
import shutil
import tempfile
from simple_idml import idml
from graph import GraphProcessingState
import pandas as pd

APP_NAME = os.getenv("APP_NAME", "gradio-test")
load_dotenv()

from graph import graph, model  # noqa

FOLLOWUP_QUESTION_NUMBER = 3
TRIM_MESSAGE_LENGTH = 16  # Includes tool messages
USER_INPUT_MAX_LENGTH = 10000  # Characters
set_verbose(True)
set_debug(True)

with open("logging-config.json", "r") as fh:
    config = json.load(fh)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


async def process_uploaded_file(file_obj, graph_state):
    """Process uploaded text file and update the graph state with its content"""
    if file_obj is None:
        return graph_state

    try:
        file_path = file_obj.name
        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "transcript" not in graph_state:
                    graph_state = dict(graph_state)
                graph_state["transcript"] = content
                return graph_state
        elif file_path.lower().endswith(".docx"):
            content = docx2txt.process(file_path)
            if "transcript" not in graph_state:
                graph_state = dict(graph_state)
            graph_state["transcript"] = content
            return graph_state
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return graph_state
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return graph_state


async def process_idml_file(file_obj, graph_state):
    """Process uploaded IDML file and update the graph state with its content"""
    if file_obj is None:
        return graph_state

    try:
        file_path = file_obj.name
        if file_path.lower().endswith(".idml"):
            # Store the IDML file path in the graph state
            if "idml_file" not in graph_state:
                graph_state = dict(graph_state)
            graph_state["idml_file"] = file_path
            return graph_state
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return graph_state
    except Exception as e:
        logger.error(f"Error processing IDML file: {str(e)}")
        return graph_state


async def chat_fn(
    user_input: str,
    history: dict,
    input_graph_state: dict,
    uuid: UUID,
    file_obj=None,
    idml_file_obj=None,
):
    """
    Args:
        user_input (str): The user's input message
        history (dict): The history of the conversation in gradio
        input_graph_state (dict): The current state of the graph. This includes tool call history
        uuid (UUID): The unique identifier for the current conversation. This can be used in conjunction with langgraph or for memory
        file_obj (file): Optional uploaded text file object
        idml_file_obj (file): Optional uploaded IDML file object
    Yields:
        str|Any: The output message
        dict|Any: The final state of the graph
        bool|Any: Whether to trigger follow up questions
        str|Any: The metrics to display
    """
    try:
        # Process the uploaded files if any
        if file_obj:
            input_graph_state = await process_uploaded_file(file_obj, input_graph_state)
        if idml_file_obj:
            input_graph_state = await process_idml_file(
                idml_file_obj, input_graph_state
            )

        if "messages" not in input_graph_state:
            input_graph_state["messages"] = []
        input_graph_state["messages"].append(
            HumanMessage(user_input[:USER_INPUT_MAX_LENGTH])
        )
        input_graph_state["messages"] = input_graph_state["messages"]

        config = RunnableConfig(
            recursion_limit=10, run_name="user_chat", configurable={"thread_id": uuid}
        )

        output: str = ""
        final_state: dict | Any = {}
        waiting_output_seq: list[str] = []

        yield "Processing...", gr.skip(), False, gr.skip()

        async for stream_mode, chunk in graph.astream(
            input_graph_state,
            config=config,
            stream_mode=["values", "messages"],
            debug=True,
        ):
            if stream_mode == "values":
                final_state = chunk
            elif stream_mode == "messages":
                msg, metadata = chunk
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for msg_tool_call in msg.tool_calls:
                        tool_name: str = msg_tool_call["name"]
                        # download_website_text is the name of the function defined in graph.py
                        if tool_name == "download_website_text":
                            waiting_output_seq.append("Downloading website text...")
                            yield "\n".join(
                                waiting_output_seq
                            ), gr.skip(), False, gr.skip()
                        elif tool_name == "tavily_search_results_json":
                            waiting_output_seq.append(
                                "Searching for relevant information..."
                            )
                            yield "\n".join(
                                waiting_output_seq
                            ), gr.skip(), False, gr.skip()
                        elif tool_name == "generate_marketing_copy":
                            waiting_output_seq.append(
                                "Generating copy from transcript..."
                            )
                            yield "\n".join(
                                waiting_output_seq
                            ), gr.skip(), False, gr.skip()
                        elif tool_name == "extract_metrics":
                            waiting_output_seq.append(
                                "Extracting metrics from transcript..."
                            )
                            yield "\n".join(
                                waiting_output_seq
                            ), gr.skip(), False, gr.skip()
                        elif tool_name:
                            waiting_output_seq.append(f"Running {tool_name}...")
                            yield "\n".join(
                                waiting_output_seq
                            ), gr.skip(), False, gr.skip()

                # print("output: ", msg, metadata)
                # assistant_node is the name we defined in the langgraph graph
                if metadata["langgraph_node"] == "assistant_node" and msg.content:
                    output += msg.content
                    yield output, gr.skip(), False, gr.skip()

        # Format metrics for display
        metrics_display = ""
        if "metrics" in final_state:
            metrics = final_state["metrics"]
            if isinstance(metrics, dict):
                metrics_display = json.dumps(metrics, indent=2)
            else:
                metrics_display = str(metrics)

        # Trigger for asking follow up questions
        # + store the graph state for next iteration
        yield output, dict(final_state), False, metrics_display
        # There's a bug in gradio where the message output isn't being fully updated before
        # The event is triggered, so try to workaround it by yielding the same output again
        yield output, gr.skip(), True, metrics_display
    except Exception:
        logger.exception("Exception occurred")
        user_error_message = (
            "There was an error processing your request. Please try again."
        )
        yield user_error_message, gr.skip(), False, gr.skip()


def clear():
    return dict(), uuid4()


class FollowupQuestions(BaseModel):
    """Model for langchain to use for structured output for followup questions"""

    questions: list[str]


async def populate_followup_questions(end_of_chat_response, messages):
    """
    This function gets called a lot due to the asynchronous nature of streaming

    Only populate followup questions if streaming has completed and the message is coming from the assistant
    """
    if not end_of_chat_response or not messages:
        return [gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)]
    if messages[-1]["role"] == "assistant":
        follow_up_questions: FollowupQuestions = await model.with_structured_output(
            FollowupQuestions
        ).ainvoke(
            [
                (
                    "system",
                    f"you are marketing assistant at a architecture firm suggest {FOLLOWUP_QUESTION_NUMBER} followup refinement for the user to ask the assistant to refine a marketing copy for a project. Refrain from asking personal questions.",
                ),
                *messages,
            ]
        )
        if len(follow_up_questions.questions) != FOLLOWUP_QUESTION_NUMBER:
            raise ValueError("Invalid value of followup questions")
        buttons = []
        for i in range(FOLLOWUP_QUESTION_NUMBER):
            buttons.append(
                gr.Button(
                    follow_up_questions.questions[i],
                    visible=True,
                    elem_classes="chat-tab",
                ),
            )
        return buttons
    else:
        return [gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)]


def click_followup_button(btn):
    buttons = [gr.Button(visible=False) for _ in range(len(followup_question_buttons))]
    return btn, *buttons


CSS = """
footer {visibility: hidden}
.followup-question-button {font-size: 12px }
.full-height-table {min-height: 70vh; max-height: 80vh; overflow-y: auto;}
"""

# We set the ChatInterface textbox id to chat-textbox for this to work
TRIGGER_CHATINTERFACE_BUTTON = """
function triggerChatButtonClick() {

  // Find the div with id "chat-textbox"
  const chatTextbox = document.getElementById("chat-textbox");

  if (!chatTextbox) {
    console.error("Error: Could not find element with id 'chat-textbox'");
    return;
  }

  // Find the button that is a descendant of the div
  const button = chatTextbox.querySelector("button");

  if (!button) {
    console.error("Error: No button found inside the chat-textbox element");
    return;
  }

  // Trigger the click event
  button.click();

}"""


def download_csv(data):
    if data is None or data.empty:
        return None
    # Create a temporary file
    temp_file = "data.csv"
    # Save DataFrame to CSV
    data.to_csv(temp_file, index=False)
    return temp_file


# following functions are for fining the placeholders and populate them with project stats and create the idml file


def find_story_files(idml_package, tag_patterns):
    """
    Find story files containing specific tags

    Args:
        idml_package: The IDML package
        tag_patterns: List of tag patterns to search for

    Returns:
        dict: Mapping of tag patterns to story files
    """
    compiled_patterns = {pattern: re.compile(pattern) for pattern in tag_patterns}
    tag_to_story = {pattern: [] for pattern in tag_patterns}
    stories = [name for name in idml_package.namelist() if name.startswith("Stories/")]

    for story_path in stories:
        try:
            content = idml_package.open(story_path).read().decode("utf-8")
            for pattern, regex in compiled_patterns.items():
                if regex.search(content):
                    tag_to_story[pattern].append(story_path)
        except Exception as e:
            print(f"Error reading {story_path}: {e}")

    return tag_to_story


def replace_content(xml_content, tag_pattern, replacements):
    """
    Replace content tags with actual data

    Args:
        xml_content: The XML content to modify
        tag_pattern: The regex pattern to match tags
        replacements: List of replacement values

    Returns:
        str: Updated XML content
    """
    tags = re.finditer(tag_pattern, xml_content)
    tag_positions = [(m.start(), m.end()) for m in tags]

    if not tag_positions:
        return xml_content

    content_chars = list(xml_content)

    for i, (start, end) in enumerate(reversed(tag_positions)):
        index = len(tag_positions) - 1 - i  # Reverse index

        if index < len(replacements):
            # Replace with actual data
            new_content = f"<Content>{replacements[index]}</Content>"
            content_chars[start:end] = new_content
        else:
            br_pattern = r"\s*<Br />"
            br_match = re.search(br_pattern, "".join(content_chars[end : end + 20]))
            if br_match:
                del content_chars[start : end + br_match.end()]
            else:
                del content_chars[start:end]

    if len(replacements) > len(tag_positions) and tag_positions:
        last_pos = tag_positions[-1][1]

        for item in replacements[len(tag_positions) :]:
            insert_content = f"\n<Content>{item}</Content>\n<Br />"
            content_chars.insert(last_pos, insert_content)
            last_pos += len(insert_content)

    return "".join(content_chars)


def create_replacements_from_metrics(metrics_data):
    """
    Convert metrics data to the replacements dictionary format

    Args:
        metrics_data: Dictionary containing project metrics

    Returns:
        dict: Mapping of tag patterns to replacement values
    """
    # Define mappings between metrics keys and IDML tag patterns
    replacements = {
        # Project Description
        r"<Content>&lt;Description&gt;</Content>": [
            metrics_data.get("description", "")
        ],
        # Project name
        r"<Content>&lt;Project Name&gt;</Content>": [
            metrics_data.get("project_name", "")
        ],
        # Location
        r"<Content>&lt;Location&gt;</Content>": [metrics_data.get("location", "")],
        # Size/Area
        r"<Content>&lt;Area&gt; SF</Content>": [metrics_data.get("size", "")],
        # Number of floors
        r"<Content>&lt;NumFloors&gt;</Content>": [
            metrics_data.get("number_of_floors", "")
        ],
        # Completion date
        r"<Content>&lt;DateComplete&gt; \(&lt;Phase&gt;\)</Content>": [
            f"{metrics_data.get('completion_date', '')}"
        ],
        # Client
        r"<Content>&lt;Client&gt;</Content>": [metrics_data.get("client_name", "")],
        # Team members - format each with a placeholder role
        r"<Content>&lt;TEAM\d+&gt; \(&lt;Role\d+&gt;\)</Content>": [
            f"{member} " for member in metrics_data.get("project_team_members", [])
        ],
        # Consultants
        r"<Content>&lt;Consultant\d+&gt;</Content>": [
            consultant for consultant in metrics_data.get("external_consultants", [])
        ],
    }

    return replacements


async def update_idml_content(idml_path, replacements_json):
    """
    Update IDML content with replacements from JSON

    Args:
        idml_path: Path to the IDML file
        replacements_json: JSON string or dict with tag patterns and replacements

    Returns:
        str: Path to the updated IDML file
    """
    # Parse JSON if it's a string
    if isinstance(replacements_json, str):
        replacements = json.loads(replacements_json)
    else:
        replacements = replacements_json

    # Get the directory where app.py is located
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a copy of the IDML file to work with
        temp_idml = os.path.join(temp_dir, "temp.idml")
        shutil.copy2(idml_path, temp_idml)

        with idml.IDMLPackage(temp_idml) as working_idml:
            # Find all story files containing our tags
            tag_patterns = list(replacements.keys())
            tag_to_story = find_story_files(working_idml, tag_patterns)

            # Extract the IDML
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            working_idml.extractall(extract_dir)

            # Process each tag pattern
            for tag_pattern, replacement_values in replacements.items():
                story_files = tag_to_story.get(tag_pattern, [])

                if not story_files:
                    print(
                        f"Warning: No story files found containing pattern '{tag_pattern}'"
                    )
                    continue

                print(
                    f"Found pattern '{tag_pattern}' in {len(story_files)} story file(s)"
                )

                # Update each story file containing this tag
                for story_path in story_files:
                    # Read the XML content
                    with open(
                        os.path.join(extract_dir, story_path), "r", encoding="utf-8"
                    ) as f:
                        xml_content = f.read()

                    # Update the content
                    updated_content = replace_content(
                        xml_content, tag_pattern, replacement_values
                    )

                    # Write back the updated content
                    with open(
                        os.path.join(extract_dir, story_path), "w", encoding="utf-8"
                    ) as f:
                        f.write(updated_content)

            # Create the output path in the same directory as app.py
            base_name = os.path.splitext(os.path.basename(idml_path))[0]
            output_filename = (
                f"{base_name}_filled_{datetime.now().strftime('%Y%m%d%H%M%S')}.idml"
            )
            output_path = os.path.join(app_dir, output_filename)

            # Create a new IDML with the updated content
            shutil.make_archive(output_path, "zip", extract_dir)
            os.rename(output_path + ".zip", output_path)

            print(f"Updated IDML saved to: {output_path}")
            return output_path


async def export_idml(graph_state: GraphProcessingState, table_data):
    """Export the current metrics, marketing copy, and table data to IDML file"""
    try:
        if "idml_file" not in graph_state:
            return None, "No IDML file uploaded"

        if "metrics" not in graph_state or "marketing_copy" not in graph_state:
            return None, "No metrics or marketing copy available"

        updated_data = dict(graph_state["metrics"])

        if table_data is not None and not table_data.empty:
            descriptions = table_data["description"].dropna().tolist()
            descriptions = [
                desc for desc in descriptions if desc.strip()
            ]  # Remove empty strings
        else:
            descriptions = [""]  # If no descriptions, create one empty file

        # Process each description and create IDML files
        output_paths = []
        import asyncio

        # Process each file one at a time to avoid race conditions
        for i, text in enumerate(descriptions):
            if "Project Description" not in text:
                updated_data["description"] = text
                print(f"Processing description {i+1}/{len(descriptions)}: {text}")
                replacements = create_replacements_from_metrics(updated_data)
                output_path = await update_idml_content(
                    graph_state["idml_file"], replacements
                )
                output_paths.append(output_path)
                # Brief pause to ensure unique timestamps
                await asyncio.sleep(1)

        print(f"Generated {len(output_paths)} IDML files: {output_paths}")
        return output_paths, f"{len(output_paths)} IDML files successfully updated"
    except Exception as e:
        import traceback

        print(f"Error in export_idml: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error updating IDML: {str(e)}"

        # Create placeholder data for the table


placeholder_data = pd.DataFrame(
    {
        "description": [
            "Project Description 1.",
            "Project Description 2.",
            "Project Description 3.",
            "Project Description 4.",
            "Project Description 5.",
        ]
    }
)
with gr.Blocks(title="Transcript to Marketing Copy", fill_height=True, css=CSS) as demo:
    uuid_state = gr.State(uuid4)
    gradio_graph_state = gr.State(dict())
    end_of_chat_response_state = gr.State(bool())

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", height=700, show_copy_button=True)
            chatbot.clear(fn=clear, outputs=[gradio_graph_state, uuid_state])

            multimodal = False
            textbox_component = gr.MultimodalTextbox if multimodal else gr.Textbox
            with gr.Row():
                followup_question_buttons = []
                for i in range(FOLLOWUP_QUESTION_NUMBER):
                    btn = gr.Button(
                        f"Button {i+1}",
                        visible=False,
                        elem_classes="followup-question-button",
                    )
                    followup_question_buttons.append(btn)
            with gr.Column():
                textbox = textbox_component(
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    autofocus=True,
                    submit_btn=True,
                    stop_btn=True,
                    elem_id="chat-textbox",
                    lines=4,
                )
            with gr.Row():
                upload_button = gr.File(
                    label="Upload Transcription File (DOCX)",
                    file_types=[".docx"],
                    scale=1,
                    height=150,
                )
                idml_upload_button = gr.File(
                    label="Upload indesign template (idml)",
                    file_types=[".idml"],
                    scale=1,
                    height=150,
                )

        with gr.Column(scale=2):

            table = gr.Dataframe(
                max_height=800,
                headers=["Description"],
                datatype=["str"],
                row_count=5,
                col_count=1,
                wrap=True,
                show_copy_button=True,
                interactive=True,
                show_row_numbers=True,
                value=placeholder_data,
            )
            with gr.Row():
                download_btn = gr.Button("Download CSV")
                export_idml_btn = gr.Button("Export to IDML")
            with gr.Row():
                idml_status = gr.Textbox(
                    label="IDML Export Status",
                    interactive=False,
                    lines=2,
                    visible=False,
                )
                idml_output = gr.File(
                    label="Download Updated IDML",
                    file_count="multiple",
                    visible=True,
                )
                download_btn.click(
                    fn=download_csv,
                    inputs=[table],
                    outputs=gr.File(label="Download CSV"),
                )
                export_idml_btn.click(
                    fn=export_idml,
                    inputs=[gradio_graph_state, table],
                    outputs=[idml_output, idml_status],
                )
            metrics_display = gr.Textbox(
                label="Project Metrics", interactive=False, lines=1, scale=1
            )

    chat_interface = gr.ChatInterface(
        chatbot=chatbot,
        fn=chat_fn,
        additional_inputs=[
            gradio_graph_state,
            uuid_state,
            upload_button,
            idml_upload_button,
        ],
        additional_outputs=[
            gradio_graph_state,
            end_of_chat_response_state,
            metrics_display,
        ],
        type="messages",
        multimodal=multimodal,
        textbox=textbox,
    )

    # for btn in followup_question_buttons:
    #     btn.click(
    #         fn=click_followup_button,
    #         inputs=[btn],
    #         outputs=[chat_interface.textbox, *followup_question_buttons],
    #     ).success(lambda: None, js=TRIGGER_CHATINTERFACE_BUTTON)

    # chatbot.change(
    #     fn=populate_followup_questions,
    #     inputs=[end_of_chat_response_state, chatbot],
    #     outputs=followup_question_buttons,
    #     trigger_mode="once",
    # )

if __name__ == "__main__":
    logger.info("Starting the interface")

    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        root_path=f"https://apps.dev.hok.com/dokku/{APP_NAME}",
    )
