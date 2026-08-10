import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from diffusers import AutoencoderKL, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from accelerate import Accelerator
from torchvision.utils import save_image

# Import existing core modules
from pipeline_sd import ADPipeline
from utils import Controller, load_image
from vae_trainer import VAETrainer

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  

# Hardware configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Global model variables
pipe = None            # AD Pipeline
control_pipe = None    # ControlNet Img2Img Pipeline

model_id = "./stable-diffusion-v1-5"  # Local root directory of SD 1.5
controlnet_dir = "./controlnet-canny"  # Local ControlNet directory
vae_path = "./trained_vae_notebook/trained_vae_shuimo" # Saved VAE path

# ------------------ Dynamic Model Resource Lifecycle Manager ------------------
def get_ad_pipeline():
    """Obtain and load the attention distillation inference pipeline (Stage 1+2)"""
    global pipe, control_pipe
    if control_pipe is not None:
        print("[Memory] Releasing ControlNet Pipeline to free up VRAM...")
        del control_pipe
        control_pipe = None
        torch.cuda.empty_cache()

    if pipe is not None:
        return pipe

    print("[System] Loading attention distillation style‑transfer pipeline...")
    if os.path.exists(vae_path):
        print(f"[System] Loading fine‑tuned VAE: {vae_path}...")
        vae = AutoencoderKL.from_pretrained(vae_path).to(device, dtype=torch.float32)
    else:
        print("[System] Fine‑tuned VAE not detected, loading standard VAE...")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float32)

    pipe = ADPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    ).to(device)

    pipe.classifier = pipe.unet 
    pipe.accelerator = Accelerator(mixed_precision="no")
    pipe.vae_scale_factor = 8
    pipe.freeze()
    return pipe

def get_control_pipeline():
    """Obtain and load ControlNet image‑to‑image pipeline (Stage 3)"""
    global pipe, control_pipe
    if pipe is not None:
        print("[Memory] Releasing attention distillation Pipeline to free up VRAM...")
        del pipe
        pipe = None
        torch.cuda.empty_cache()

    if control_pipe is not None:
        return control_pipe

    if not os.path.exists(controlnet_dir):
        raise FileNotFoundError(f"Local ControlNet Canny weight directory not found: {controlnet_dir}")

    print("[System] Loading ControlNet Canny condition controller...")
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch_dtype)

    print("[System] Loading Stable Diffusion ControlNet image‑to‑image pipeline...")
    control_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
        local_files_only=True
    ).to(device)

    if device == "cuda":
        control_pipe.enable_attention_slicing()
    return control_pipe


@app.before_request
def prepare_model():
    # Default warm‑start loading
    global pipe
    if pipe is None and control_pipe is None:
        get_ad_pipeline()

@app.route('/')
def index():
    return render_template('index.html')


# ================= 1. Attention Distillation Pipeline Route (Stage 1+2) =================
@app.route('/transfer', methods=['POST'])
def run_transfer():
    global pipe, vae_path
    try:
        content_file = request.files.get('content_img')
        style_file = request.files.get('style_img')
        
        if not content_file or not style_file:
            return jsonify({"status": "error", "message": "Please ensure both content image and style image are uploaded!"}), 400

        # Retrieve parameters
        iters = int(request.form.get('iters', 200))
        weight = float(request.form.get('weight', 0.25))
        lr = float(request.form.get('lr', 0.05))

        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content_' + content_filename)
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style_' + style_filename)
        
        content_file.save(content_path)
        style_file.save(style_path)

        # ------------------ Step 1: VAE decoder fine‑tuning ------------------
        print("\n=== [Stage 1] Starting VAE fine‑tuning stage ===")
        if pipe is not None:
            del pipe
            pipe = None
            torch.cuda.empty_cache()
            
        trainer = VAETrainer(
            base_model_path=model_id,
            lr=1e-4,
            epochs=75,
            device=device
        )
        trained_vae_dir = "./trained_vae_notebook"
        saved_vae_path = trainer.train(style_path, out_dir=trained_vae_dir)
        vae_path = saved_vae_path

        # ------------------ Step 2: Attention Distillation feature transfer ------------------
        print("\n=== [Stage 2] Starting attention distillation feature transfer stage ===")
        get_ad_pipeline()

        content_image = load_image(content_path, size=(512, 512)).to(device)
        style_image = load_image(style_path, size=(512, 512)).to(device)

        controller = Controller(self_layers=(10, 16)) 

        result = pipe.optimize(
            style_image=style_image,
            content_image=content_image,
            controller=controller,
            lr=lr,
            iters=iters,
            weight=weight,
            num_inference_steps=50,
            width=512,
            height=512
        )

        output_filename = f"out_{iters}_{weight}_{content_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        save_image(result[0].cpu(), output_path)

        return jsonify({
            "status": "success",
            "result_url": f"/{output_path}"
        })

    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Error occurred during processing: {str(e)}"}), 500


# ================= 2.  ControlNet with Canny Edge Detection (Stage 3) =================
@app.route('/refine', methods=['POST'])
def run_refine():
    global control_pipe
    try:
        # Support two modes: user‑uploaded source image, or locally saved image generated from Stage 2
        refine_file = request.files.get('refine_img')
        local_refine_path = request.form.get('local_refine_path')

        if not refine_file and not local_refine_path:
            return jsonify({"status": "error", "message": "No input source detected! Please generate an image first or upload a base image."}), 400

        # Retrieve refinement control parameters
        style_preservation = float(request.form.get('style_preservation', 0.75))
        neg_space_val = int(request.form.get('negative_space', 50))
        ink_depth = int(request.form.get('ink_depth', 50))
        brush_val = int(request.form.get('brush_wetness', 50))

        # Locate the file path of the base image for processing
        if refine_file:
            filename = secure_filename(refine_file.filename)
            input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'refine_src_' + filename)
            refine_file.save(input_img_path)
        else:
            # Strip leading "/" to avoid path lookup exceptions
            input_img_path = local_refine_path.lstrip('/')

        #----------- Pre‑processing: Physical image pre‑processing for contrast adjustment--------------
        print("\n=== [Stage 3] Starting physical pre‑processing, adjusting ink tone and brush strokes...")
        init_image = Image.open(input_img_path).convert("RGB").resize((512, 512))
        image_np = np.array(init_image)

        # 1. Ink density adjustment via brightness / contrast tuning
        brightness_factor = 1.8 - (ink_depth / 100.0) * 1.5  
        contrast_factor = 0.6 + (ink_depth / 100.0) * 1.4    
        
        enhancer_b = ImageEnhance.Brightness(init_image)
        init_image = enhancer_b.enhance(brightness_factor)
        enhancer_c = ImageEnhance.Contrast(init_image)
        init_image = enhancer_c.enhance(contrast_factor)
        
        if ink_depth < 20:
            grey_layer = Image.new('RGB', init_image.size, (180, 180, 180))
            init_image = Image.blend(init_image, grey_layer, 0.4)

        # 2. Negative‑space colour adjustment 
        saturation_factor = 1.2 - (neg_space_val / 100.0) * 1.1
        enhancer_s = ImageEnhance.Color(init_image)
        init_image = enhancer_s.enhance(max(0.1, saturation_factor))

        # 3. Brush wet‑dry effect
        canny_edges = cv2.Canny(image_np, 100, 200)
        
        if brush_val >= 80:  
            kernel = np.ones((1, 1), np.uint8)
            canny_edges = cv2.erode(canny_edges, kernel, iterations=1)
            noise = np.random.randint(0, 255, canny_edges.shape, dtype=np.uint8)
            noise_mask = (noise > 230)
            canny_edges[noise_mask] = 255
        elif brush_val <= 20:  
            kernel = np.ones((7, 7), np.uint8)
            canny_edges = cv2.dilate(canny_edges, kernel, iterations=2)
            canny_edges = cv2.GaussianBlur(canny_edges, (15, 15), 0)
        else:  
            canny_edges = cv2.GaussianBlur(canny_edges, (3, 3), 0)
            
        canny_edges = np.concatenate([canny_edges[:, :, None]] * 3, axis=2)
        canny_image = Image.fromarray(canny_edges)

        # ------------------ Dynamic inference component tuning ------------------
        # Brush‑stroke control strength
        if brush_val >= 80:
            ctrl_scale = 1.2
        elif brush_val <= 20:
            ctrl_scale = 0.6
        else:
            ctrl_scale = 0.9
            
        # Negative‑space control strength
        guidance = 7.0 + (neg_space_val / 100.0) * 4.0  

        # Dynamic maximum repainting gain, unlock extreme contrast range
        ink_dev = abs(ink_depth - 50) / 50.0   
        brush_dev = abs(brush_val - 50) / 50.0 
        max_dev = max(ink_dev, brush_dev)
        
        base_strength = round(1.0 - style_preservation, 2)
        strength_boost = max_dev * 0.15
        final_strength = min(0.90, round(base_strength + strength_boost, 2))

        # ------------------ Assemble prompt framework ------------------
        if ink_depth >= 80:
            ink_prompt = "(saturated pitch-black carbon ink:1.5), (heavy dark solid black ink washes:1.4), high contrast"
        elif ink_depth <= 20:
            ink_prompt = "(faint translucent light-grey ink washes:1.5), (pale misty translucent ink:1.4), very low contrast"
        else:
            ink_prompt = "balanced traditional black and grey ink washes"

        if brush_val >= 80:
            brush_prompt = "(extremely dry brush strokes:1.5), (sharp calligraphic feibai strokes:1.4), visible split brush hairs"
        elif brush_val <= 20:
            brush_prompt = "(wet splash-ink washes:1.5), (soft watercolor-style ink bleeding:1.4), blooming ink edges"
        else:
            brush_prompt = "expressive traditional Chinese brushwork"

        if neg_space_val >= 70:
            space_prompt = "abundant negative space, minimalist composition, spacious empty background blending with paper"
        elif neg_space_val <= 30:
            space_prompt = "dense composition, intricate detailed background, rich ink wash textures"
        else:
            space_prompt = "natural balanced composition with elegant negative space"

        prompt = f"traditional Chinese ink wash painting masterwork, {ink_prompt}, {brush_prompt}, {space_prompt}, fine art style"
        negative_prompt = "colorful, vibrant, 3d, realistic, photorealistic, modern, bad quality, frames, borders, signatures, stamp"

        # ------------------ Launch ControlNet inference ------------------
        print("=== [Stage 3] Loading and executing ControlNet brush‑stroke reshaping...")
        get_control_pipeline()

        with torch.inference_mode():
            result = control_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                control_image=canny_image,
                strength=final_strength,
                controlnet_conditioning_scale=ctrl_scale,
                guidance_scale=guidance,
                num_inference_steps=35
            ).images[0]

        # Export and save the final output image
        out_refine_filename = f"refined_out_{brush_val}_{ink_depth}_{secure_filename(os.path.basename(input_img_path))}"
        out_refine_path = os.path.join(app.config['OUTPUT_FOLDER'], out_refine_filename)
        result.save(out_refine_path)

        print(f"Refinement completed successfully, saved to: {out_refine_path}")
        return jsonify({
            "status": "success",
            "result_url": f"/{out_refine_path}"
        })

    except Exception as e:
        print(f"Refinement failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Error during refinement processing: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)