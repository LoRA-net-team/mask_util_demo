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
    # score array is 0 to 1 float array(1d)
    # convert to grayscale
    binary_mask = mask.convert('L') # between 0 and 255
    print("binary_mask", binary_mask.size)
    score_arr = load_numpy_array(score_path, binary_mask.size)
    # if dim is 3, mean is taken
    if len(score_arr.shape) == 3:
        score_arr = score_arr.mean(axis=2)
    print("score_arr", score_arr.shape)
    if smoothen_array:
        binary_mask = linear_smoothing_filter(binary_mask, kernel_size) # between 0 and 255
        # convert to image for preview
    preview_arr = np.array(binary_mask)
    # convert to image for preview, round to int
    preview_arr = preview_arr.round().astype(np.uint8)
    # apply mask, convert to float with 2-dim
    float_mask_arr = np.array(binary_mask) / 255
    masked_score_arr = score_arr * float_mask_arr
    score_outside = score_arr * (1 - float_mask_arr)
    # calculate sum
    masked_sum = masked_score_arr.sum()
    score_outside_sum = score_outside.sum()
    percentage = masked_sum / (masked_sum + score_outside_sum) * 100
    return masked_sum, score_outside_sum,percentage, preview_arr

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
            mask_input = gr.Image(label="Mask", type="pil", source="upload") # disable interactive
            score_path_input = gr.Textbox(lines=1, label="Score array path")
            smoothen_array_input = gr.Checkbox(label="Smoothen array", default=False)
            kernel_size_input = gr.Slider(minimum=1, maximum=100, step=1, default=1, label="Kernel size")
            preview_arr_output = gr.outputs.Image(type="numpy", label="Preview array")
            masked_sum_output = gr.Textbox(lines=1, label="Masked sum score")
            score_outside_sum_output = gr.Textbox(lines=1, label="Score outside sum")
            score_percentage_output = gr.Textbox(lines=1, label="Score percentage")
            masked_sum_button = gr.Button("Submit")
            masked_sum_button.click(masked_sum_score, inputs=[mask_input, score_path_input, smoothen_array_input, kernel_size_input], outputs=[masked_sum_output,score_outside_sum_output,score_percentage_output, preview_arr_output])

blocks.launch()
