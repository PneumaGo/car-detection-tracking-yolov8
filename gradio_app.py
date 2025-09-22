import gradio as gr
import os
from .image_processor import process_image
from .video_processor import process_video_and_count

def gradio_process_media(media_file, progress=gr.Progress()):
    if media_file is None:
        return None, "Please upload an image or video file."

    file_extension = os.path.splitext(media_file.name)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        return process_image(media_file.name)
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return process_video_and_count(media_file.name, progress)
    else:
        return None, "Unsupported file type. Please upload an image or video."

def launch_interface():
    input_component = gr.File(label="Upload an image or video file", file_types=["image", "video"])
    output_media = gr.Video(label="Annotated Output")
    output_text = gr.Textbox(label="Analysis Results")

    iface = gr.Interface(
        fn=gradio_process_media,
        inputs=input_component,
        outputs=[output_media, output_text],
        title="YOLOv8 Demo: Object Detection, Tracking, and Counting",
        description="Upload an image for detection or a video for detection, tracking, and counting.",
        allow_flagging="never",
        examples=[["test_video.mp4"], ["test_image.jpg"]]
    )
    iface.queue().launch(share=True)
