import gradio as gr
from PIL import Image
from typing import Union
import numpy as np
import cv2
# create interface with mask for drawing


blocks = gr.Blocks()

def binary_mask(image:Union[Image.Image, dict]) -> Image.Image:
    # convert to grayscale
    if isinstance(image, dict):
        image = image['mask']
    gray = image.convert('L')
    gray = np.array(gray)
    # threshold to get a mask
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask = Image.fromarray(mask)
    return mask

with blocks:
    with gr.Tabs():
        # first tab is for drawing mask
        with gr.TabItem("Draw mask"):
            image_input = gr.Image(label="Input Image", type="pil", tool="sketch", source="upload", interactive=True)
            mask_output = gr.outputs.Image(type="pil")
            mask_button = gr.Button("Submit")
            mask_button.click(binary_mask, inputs=[image_input], outputs=[mask_output])
            
blocks.launch()