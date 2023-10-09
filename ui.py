import gradio as gr
from PIL import Image
from typing import Union, Tuple
import numpy as np
import cv2
# create interface with mask for drawing
blocks = gr.Blocks()

def binary_mask(image:Union[Image.Image, dict]) -> Image.Image:
    """
    Convert image to binary mask
    """
    # convert to grayscale
    if isinstance(image, dict):
        image = image['mask']
    gray = image.convert('L')
    gray = np.array(gray)
    # threshold to get a mask
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask = Image.fromarray(mask)
    return mask

def linear_smoothing_filter(image:Image.Image, kernel_size:int) -> Image.Image:
    """
    Apply linear smoothing filter to image
    """
    # convert to grayscale
    gray = image.convert('L')
    gray = np.array(gray)
    # apply linear smoothing filter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    smoothed = cv2.filter2D(gray, -1, kernel)
    smoothed = Image.fromarray(smoothed)
    return smoothed

def load_numpy_array(filepath:str, target_res:Tuple[int, int]) -> np.ndarray:
    """
    Load npy file from filepath, resize to target_res
    The total sum of the array is 1. Resized array will also have total sum of 1.
    """
    # load npy file
    array = np.load(filepath)
    # resize to target_res
    array = cv2.resize(array, target_res)
    # normalize to sum of 1
    assert array.sum() > 0, "array should have sum > 0"
    array = array / array.sum()
    return array

def masked_sum_score(mask:Image.Image, score_path:str, smoothen_array:bool, kernel_size:int) -> float:
    """
    Calculate masked sum score from mask and score_path.
    Score array will be resized to mask size.
    The score is always between 0 and 1.
    
    @param mask: mask image
    @param score_path: path to score array
    @param smoothen_array: whether to apply linear smoothing filter to score array
    @param kernel_size: kernel size for linear smoothing filter
    """
    score_arr = load_numpy_array(score_path, mask.size)
    if smoothen_array:
        score_arr = linear_smoothing_filter(Image.fromarray(score_arr), kernel_size)
        score_arr = np.array(score_arr)
    # apply mask
    mask_arr = np.array(mask)
    masked_score_arr = score_arr * mask_arr
    # calculate sum
    masked_sum = masked_score_arr.sum()
    return masked_sum

# create interface
with blocks:
    with gr.Tabs():
        # first tab is for drawing mask
        with gr.TabItem("Draw mask"):
            image_input = gr.Image(label="Input Image", type="pil", tool="sketch", source="upload", interactive=True)
            mask_output = gr.outputs.Image(type="pil")
            mask_button = gr.Button("Submit")
            mask_button.click(binary_mask, inputs=[image_input], outputs=[mask_output])
        # second tab is for calculating masked sum score
        with gr.TabItem("Masked sum score"):
            mask_input = gr.Image(label="Mask", type="pil", source="upload", interactive=False) # disable interactive
            score_path_input = gr.Textbox(lines=1, label="Score array path")
            smoothen_array_input = gr.Checkbox(label="Smoothen array", default=False)
            kernel_size_input = gr.Slider(minimum=1, maximum=100, step=1, default=1, label="Kernel size")
            masked_sum_output = gr.Textbox(lines=1)
            masked_sum_button = gr.Button("Submit")
            masked_sum_button.click(masked_sum_score, inputs=[mask_input, score_path_input, smoothen_array_input, kernel_size_input], outputs=[masked_sum_output])

blocks.launch()
