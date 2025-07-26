import cv2
import numpy as np
from PIL import Image, ImageEnhance
import gradio as gr
from gradio_imageslider import ImageSlider
import tempfile
import os
from pathlib import Path

def find_white_reference(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, 95)
    bright_mask = gray > threshold
    bright_pixels = image[bright_mask]

    if len(bright_pixels) > 100:
        white_ref = np.median(bright_pixels, axis=0)
        return white_ref, True
    else:
        return None, False

def apply_white_balance(image, white_balance_strength):
    white_ref, has_reference = find_white_reference(image)
    img_float = image.astype(np.float32)

    if has_reference:
        target_white = max(white_ref)
        scale_b = target_white / (white_ref[0] + 1e-6)
        scale_g = target_white / (white_ref[1] + 1e-6)
        scale_r = target_white / (white_ref[2] + 1e-6)
        
        # Apply strength factor
        scale_b = 1.0 + (scale_b - 1.0) * white_balance_strength
        scale_g = 1.0 + (scale_g - 1.0) * white_balance_strength
        scale_r = 1.0 + (scale_r - 1.0) * white_balance_strength
        
        scale_b = np.clip(scale_b, 0.6, 2.0)
        scale_g = np.clip(scale_g, 0.6, 2.0)
        scale_r = np.clip(scale_r, 0.6, 2.0)
    else:
        mean_b = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_r = np.mean(img_float[:, :, 2])
        target = max(mean_b, mean_g, mean_r)
        scale_b = target / (mean_b + 1e-6)
        scale_g = target / (mean_g + 1e-6)
        scale_r = target / (mean_r + 1e-6)
        
        # Apply strength factor
        scale_b = 1.0 + (scale_b - 1.0) * white_balance_strength
        scale_g = 1.0 + (scale_g - 1.0) * white_balance_strength
        scale_r = 1.0 + (scale_r - 1.0) * white_balance_strength
        
        scale_b = np.clip(scale_b, 0.7, 1.8)
        scale_g = np.clip(scale_g, 0.7, 1.8)
        scale_r = np.clip(scale_r, 0.7, 1.8)

    img_float[:, :, 0] *= scale_b
    img_float[:, :, 1] *= scale_g
    img_float[:, :, 2] *= scale_r

    return np.clip(img_float, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image, target_brightness, contrast_strength):
    current_brightness = np.mean(image.astype(np.float32) / 255.0)
    brightness_boost = target_brightness / (current_brightness + 1e-6)
    brightness_boost = np.clip(brightness_boost, 0.8, 2.0)

    img_float = image.astype(np.float32) / 255.0

    if brightness_boost > 1.0:
        gamma = 1.0 / (1.0 + (brightness_boost - 1.0) * 0.8)
        img_boosted = np.power(img_float, gamma)
    else:
        img_boosted = img_float * brightness_boost

    lab = cv2.cvtColor((img_boosted * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply contrast strength
    clahe_clip = 1.0 + 2.0 * contrast_strength
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    current_contrast = np.std(l)
    blend_factor = (0.7 if current_contrast < 30 else 0.4) * contrast_strength
    l_final = ((1 - blend_factor) * l + blend_factor * l_enhanced).astype(np.uint8)

    lab_final = cv2.merge([l_final, a, b])
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

def remove_color_cast(image, cast_strength):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    a_mean = np.mean(a) - 128
    b_mean = np.mean(b) - 128

    a_corrected = a.astype(np.float32)
    b_corrected = b.astype(np.float32)

    if abs(a_mean) > 2:
        a_correction = -a_mean * 0.8 * cast_strength
        a_corrected += a_correction

    if abs(b_mean) > 2:
        b_correction = -b_mean * 0.8 * cast_strength
        b_corrected += b_correction

    a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    lab_corrected = cv2.merge([l, a_corrected, b_corrected])

    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

def final_enhance(image, original_brightness, saturation_boost, sharpness_boost):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    pil_image = ImageEnhance.Color(pil_image).enhance(saturation_boost)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness_boost)

    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    final_brightness = np.mean(result.astype(np.float32) / 255.0)

    if final_brightness < original_brightness * 0.9:
        brightness_multiplier = (original_brightness / final_brightness) * 1.05
        brightness_multiplier = min(brightness_multiplier, 1.2)
        result_float = result.astype(np.float32) * brightness_multiplier
        result = np.clip(result_float, 0, 255).astype(np.uint8)

    return result

# Global variable to store the processed image
processed_image = None

def process_image(image, white_balance_strength, cast_strength, brightness_boost, 
                 contrast_strength, saturation_boost, sharpness_boost):
    global processed_image
    
    if image is None:
        processed_image = None
        return None
        
    # Convert PIL to OpenCV with high quality
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    original_brightness = np.mean(image_cv.astype(np.float32) / 255.0)

    corrected = apply_white_balance(image_cv, white_balance_strength)
    corrected = remove_color_cast(corrected, cast_strength)
    target_brightness = original_brightness * brightness_boost
    corrected = adjust_brightness_contrast(corrected, target_brightness, contrast_strength)
    corrected = final_enhance(corrected, original_brightness, saturation_boost, sharpness_boost)
    
    # Convert back to PIL with high quality
    corrected_pil = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    
    # Store the processed image for download
    processed_image = corrected_pil
    
    return (image, corrected_pil)

def download_image():
    """Function to handle image download"""
    if processed_image is None:
        return None
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    processed_image.save(temp_file.name, 'PNG', quality=95)
    temp_file.close()
    
    return temp_file.name

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# ChatGPT Yellow Tint Remover")
    gr.Markdown("This is a yellow tint remover tool designed to automatically remove yellow and other unwated color tints from images generated by ChatGPT.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil")
            
            with gr.Accordion("Settings", open=False):
                white_balance_strength = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.8, step=0.1,
                    label="White Balance"
                )
                cast_strength = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.8, step=0.1,
                    label="Color Cast Removal"
                )
                brightness_boost = gr.Slider(
                    minimum=0.8, maximum=1.3, value=1.1, step=0.05,
                    label="Brightness"
                )
                contrast_strength = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                    label="Contrast"
                )
                saturation_boost = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.1, step=0.1,
                    label="Saturation"
                )
                sharpness_boost = gr.Slider(
                    minimum=0.5, maximum=1.5, value=1.05, step=0.05,
                    label="Sharpness"
                )
                
                # Reset button
                reset_btn = gr.Button("Reset to Defaults")
            
        with gr.Column(scale=2):
            output_slider = ImageSlider(label="Before / After", type="pil", interactive=True)
            
            # Download button positioned right under the image slider
            download_btn = gr.DownloadButton(
                label="Download",
                value=None,
                variant="primary"
            )

    # Auto-process when image is uploaded or settings change
    input_image.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    # Process when any setting changes
    white_balance_strength.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    cast_strength.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    brightness_boost.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    contrast_strength.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    saturation_boost.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    sharpness_boost.change(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )
    
    # Reset to defaults
    reset_btn.click(
        fn=lambda: [0.8, 0.8, 1.1, 0.7, 1.1, 1.05],
        outputs=[white_balance_strength, cast_strength, brightness_boost, 
                contrast_strength, saturation_boost, sharpness_boost]
    ).then(
        fn=process_image,
        inputs=[input_image, white_balance_strength, cast_strength, brightness_boost, 
               contrast_strength, saturation_boost, sharpness_boost],
        outputs=output_slider
    ).then(
        fn=download_image,
        outputs=download_btn
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)
