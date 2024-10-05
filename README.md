# Background Remover

This project is a background remover tool that utilizes a pre-trained model from Hugging Face for image segmentation and the Gradio library to create an easy-to-use web interface. Users can upload an image, and the model will automatically remove its background.

## 📋 Project Description

The main goal of this project is to provide a simple and effective way to remove the background from images using machine learning. The Hugging Face `image-segmentation` pipeline is used to process the image and remove its background. The Gradio library is used to build a user-friendly interface for uploading and processing images.

## 🛠️ Technologies Used

- **Language**: Python
- **Libraries**:
  - `huggingface transformers`: To load and use the pre-trained model for image segmentation
  - `gradio`: To create the interactive user interface for the background remover

## ⚙️ Implementation

### Background Removal Function

The function `remove_background` takes an image as input and processes it using the Hugging Face `image-segmentation` pipeline with the model `"briaai/RMBG-1.4"`. The model returns the segmented image without its background. Here's a breakdown of the process:
- **Pipeline**: The Hugging Face `pipeline` function loads the pre-trained model for image segmentation.
- **Image Processing**: The model generates a mask and removes the background from the input image.

### Gradio Interface

The Gradio interface allows users to upload an image, which is then passed to the `remove_background` function. The output is the same image with the background removed. The interface includes:
- **Inputs**: An image uploader (`gr.components.Image`) allowing users to submit images in `.png`, `.jpg`, or `.jpeg` formats.
- **Outputs**: The processed image with the background removed, displayed in the Gradio interface.

### Code Overview

```python
from transformers import pipeline
import gradio as gr

def remove_background(image):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(image, return_mask=True)
    pillow_image = pipe(image)
    
    return pillow_image

app = gr.Interface(
    fn=remove_background,
    inputs=gr.components.Image(type="pil"),
    outputs=gr.components.Image(type="pil"),
    title="Background Remover",
    description="Upload an image and remove its background"
)

app.launch(share=True)
```

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/AykoSousa/background_remover.git
   ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

The project is running on my hugging face workspace 😁
- https://huggingface.co/spaces/aykoNasc1men7o/background_remover