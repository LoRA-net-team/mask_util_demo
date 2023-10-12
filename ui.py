import gradio as gr
from PIL import Image
from typing import Union, Tuple, Dict
import numpy as np
import cv2
import os
from seaborn import heatmap
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
    #if mask is str, open path
    if isinstance(mask, str):
        if not os.path.exists(mask):
            raise FileNotFoundError(f"mask file does not exist: {mask}")
        mask = Image.open(mask)
    # if mask is 3-channel, match black area
    binary_mask = mask.convert('L') # between 0 and 255
    print("binary_mask", binary_mask.size)
    score_arr = load_numpy_array(score_path, binary_mask.size)
    # if dim is 3, mean is taken
    if len(score_arr.shape) == 3:
        score_arr = score_arr.mean(0)
    print("score_arr", score_arr.shape)
    # apply threshold to binary mask
    binary_mask = binary_mask.point(lambda x: 255 if x < 25 else 0)
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
    
    #heatmap
    score_heatmap = heatmap(score_arr, cmap="YlGnBu")
    # convert to image for preview
    buffer:memoryview = score_heatmap.get_figure().canvas.buffer_rgba()
    score_heatmap_pil = Image.frombuffer('RGBA', score_heatmap.get_figure().canvas.get_width_height(), buffer, 'raw', 'RGBA', 0, 1)
    return masked_sum, score_outside_sum,percentage, preview_arr, score_heatmap_pil

def preprocess_mask(mask:Image.Image) -> Image.Image:
    """
    Preprocess mask image.
    Mask may be arbitrary 3-channel image
    """
    mask_processed = mask.convert('L') # convert to grayscale
    # apply threshold
    mask_processed = mask_processed.point(lambda x: 255 if x < 25 else 0)
    return mask_processed


def calculate_folder(folder_path:str, kernel_size:int, smoothen_array:bool) -> Dict[str, float]:
    """
    Calculate masked sum score for all images in folder_path.
    image with _mask.png will be used as mask.
    image with *.npy will be used as score array.
    
    @return: dict of {score_file: score_percentage}
    
    @throws FileNotFoundError: if folder_path does not exist, or mask_file or score_files are not found
    """
    # check if folder_path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"folder_path does not exist: {folder_path}")
    # get all files in folder_path
    files = os.listdir(folder_path)
    # get mask file and score file
    mask_file = None
    score_files = []
    for file in files:
        if file.endswith("_mask.png"):
            mask_file = file
        elif file.endswith(".npy"):
            score_files.append(file)
    # check if mask_file and score_files are found
    if mask_file is None:
        raise FileNotFoundError(f"mask_file not found in folder_path: {folder_path}")
    if len(score_files) == 0:
        raise FileNotFoundError(f"score_files not found in folder_path: {folder_path}")
    # calculate score for each score file
    scores = {}
    for score_file in score_files:
        # get score path
        score_path = os.path.join(folder_path, score_file)
        mask_file_path = os.path.join(folder_path, mask_file)
        # get masked sum score
        masked_sum, score_outside_sum, percentage, preview_arr, heatmap_arr = masked_sum_score(mask_file_path, score_path, smoothen_array, kernel_size)
        # add to scores
        scores[score_file] = percentage
    return scores
        
def calculate_folder_recursive(folder_path:str, kernel_size:int, smoothen_array:bool, ignore_error:bool=False) -> Dict[str, Dict[str, float]]:
    """
    Calculate folders recursively.
    """
    # get all folders in folder_path
    folders = os.listdir(folder_path)
    # skip empty folders
    if len(folders) == 0:
        return {}
    # skip folders with no .npy files
    if len([f for f in folders if f.endswith(".npy")]) == 0:
        return {}
    scores = {}
    for folder in folders:
        # check if folder is a folder
        folder = os.path.join(folder_path, folder)
        if not os.path.isdir(folder):
            continue
        # if folder is empty, skip
        if len(os.listdir(folder)) == 0:
            continue
        # calculate score for folder
        try:
            scores[folder] = calculate_folder(folder, kernel_size, smoothen_array)
        except FileNotFoundError as e:
            if ignore_error:
                print(f"Error in folder: {folder}")
                print(e)
            else:
                raise e
    # return scores
    return scores



# create interface
with blocks:
    with gr.Tabs():
        # first tab is for drawing mask
        with gr.TabItem("Draw mask"):
            image_input = gr.Image(label="Input Image", type="pil", tool="sketch", source="upload", interactive=True)
            mask_output = gr.outputs.Image(type="pil")
            mask_button = gr.Button("Submit")
            mask_button.click(binary_mask, inputs=[image_input], outputs=[mask_output])
        with gr.TabItem("Mask preview"):
            mask_input_2 = gr.Image(label="Mask", type="pil", source="upload")
            preview_mask_output = gr.outputs.Image(type="pil")
            preprocess_button = gr.Button("Submit")
            preprocess_button.click(preprocess_mask, inputs=[mask_input_2], outputs=[preview_mask_output])
        # second tab is for calculating masked sum score
        with gr.TabItem("Masked sum score"):
            mask_input = gr.Image(label="Mask", type="pil", source="upload")
            score_path_input = gr.Textbox(lines=1, label="Score array path")
            smoothen_array_input = gr.Checkbox(label="Smoothen array", default=False)
            kernel_size_input = gr.Slider(minimum=1, maximum=100, step=1, default=1, label="Kernel size")
            preview_arr_output = gr.outputs.Image(type="numpy", label="Preview array")
            masked_sum_output = gr.Textbox(lines=1, label="Masked sum score")
            score_outside_sum_output = gr.Textbox(lines=1, label="Score outside sum")
            score_percentage_output = gr.Textbox(lines=1, label="Score percentage")
            score_visualization_output = gr.outputs.Image(type="pil", label="Score visualization")
            masked_sum_button = gr.Button("Submit")
            masked_sum_button.click(masked_sum_score, inputs=[mask_input, score_path_input, smoothen_array_input, kernel_size_input], outputs=[masked_sum_output,score_outside_sum_output,score_percentage_output, preview_arr_output, score_visualization_output])
        with gr.TabItem("Masked sum score for folders"):
            folder_path_input = gr.Textbox(lines=1, label="Folder path")
            smoothen_array_input = gr.Checkbox(label="Smoothen array", default=False)
            kernel_size_input = gr.Slider(minimum=1, maximum=100, step=1, default=1, label="Kernel size")
            ignore_error_input = gr.Checkbox(label="Ignore error", default=False)
            scores_output = gr.outputs.Textbox(label="Scores")
            calculate_folder_button = gr.Button("Submit")
            calculate_folder_button.click(calculate_folder_recursive, inputs=[folder_path_input, kernel_size_input, smoothen_array_input, ignore_error_input], outputs=[scores_output])
blocks.launch()
