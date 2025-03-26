import gradio as gr
import os

APP_NAME = os.getenv("APP_NAME", "gradio-test")

def greet(name, intensity):
    return "Hello! " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(server_name="0.0.0.0", server_port=5000, root_path=f"https://apps.dev.hok.com/dokku/{APP_NAME}")
