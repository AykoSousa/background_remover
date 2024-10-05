# Import libs
from transformers import pipeline
import gradio as gr

# Define function that's instance and remove the background
def remove_background(image):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(image, return_mask = True)
    pillow_image = pipe(image)

    return pillow_image

# build the frontend with gradio
app = gr.Interface(
    fn = remove_background,
    inputs = gr.components.Image(type = "pil"),
    outputs = gr.components.Image(type = "pil"),
    title = "Background Remover",
    description = "Upload an image and remove its background"
)

if __name__ == __name__:
    # Option share = True anable a temp link to share with another peoples
    app.launch(share=True)