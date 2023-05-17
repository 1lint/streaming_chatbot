import gradio as gr
import time
import openai
from pathlib import Path


def transcribe(audio, chatbot_history, openai_key):
    time.sleep(5)
    transcript = openai.Audio.transcribe("whisper-1", open(audio, "rb"), api_key=openai_key)

    content = transcript["text"]
    if content:
        if not chatbot_history:
            return [[content, None]]
        else:
            return chatbot_history + [[content, None]]
    else:
        return chatbot_history



def openai_stream(history, openai_key, chat_model):

    if not history or history[-1][1]:
        return history

    history[-1][1] = ""
    for chunk in openai.ChatCompletion.create(
        model=chat_model,
        messages=[{
            "role": "user",
            "content": history[-1][0]
        }],
        stream=True,
        api_key=openai_key,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content:
            history[-1][1] += content
            yield history

    

def show_message(user_message, history):
    return "", history + [[user_message, None]]

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

parent_path = Path(__file__).parent

with open(parent_path / "header.MD") as fp:
    header = fp.read()

available_models = ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']

with gr.Blocks(theme=theme) as demo:


    header_component = gr.Markdown(header)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="Talk with ChatGPT", source="microphone", type="filepath", streaming=True)
            clear = gr.Button("Clear Chat History")
            dark_mode_btn = gr.Button("Dark Mode", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                chat_model = gr.Dropdown(choices=available_models, value="gpt-3.5-turbo", allow_custom_value=True)
                openai_key = gr.Textbox(label="Enter OPENAI API Key", placeholder="Example: sk-AJDKakdAJD...")
            chatbot = gr.Chatbot(label="ChatGPT Dialog")
            msg = gr.Textbox(label="Chat with ChatGPT", placeholder="Press <Enter> to submit")

    streaming_event_kwargs = dict(
        fn=openai_stream, 
        inputs=[chatbot, openai_key, chat_model], 
        outputs=chatbot,
    )


    msg.submit(show_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        **streaming_event_kwargs
    )
    audio.stream(transcribe, inputs=[audio, chatbot, openai_key], outputs=[chatbot]).then(
       **streaming_event_kwargs
    )

    clear.click(lambda: None, None, chatbot, queue=False)

    # from gradio.themes.builder
    toggle_dark_mode_args = dict(
        fn=None,
        inputs=None,
        outputs=None,
        _js="""() => {
        if (document.querySelectorAll('.dark').length) {
                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
            } else {
                document.querySelector('body').classList.add('dark');
            }
        }""",
    )
    demo.load(**toggle_dark_mode_args)
    dark_mode_btn.click(**toggle_dark_mode_args)


demo.queue()
demo.launch()

