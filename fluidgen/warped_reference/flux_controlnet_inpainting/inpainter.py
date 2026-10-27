
import os
import sys
from pathlib import Path

import torch
from flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
from flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel
from flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fluidgen import paths  # noqa: E402

def dilate_binary_mask(mask: np.ndarray, size=(512, 512), kernel_size=5, iterations=1):
    """
    Args:
        mask: np.ndarray, binary mask with values 0 or 1
        size: output image size (W, H)
        kernel_size: dilation kernel size (odd int)
        iterations: number of dilation iterations
    Returns:
        PIL.Image: dilated binary mask as RGB image with values 0 or 255
    """
    # Ensure mask is binary uint8 (0 or 255)
    mask = mask.detach().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Resize to target size
    mask_resized = cv2.resize(mask_uint8, size, interpolation=cv2.INTER_NEAREST)

    # Apply dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_resized, kernel, iterations=iterations)

    # Convert to RGB and ensure it's still binary (0 or 255)
    dilated_binary = np.where(dilated > 0, 255, 0).astype(np.uint8)

    return dilated_binary

def smooth_segmentation_mask_255(
    mask: np.ndarray,
    blur_kernel_size: int = 15,
    blur_sigma: float = 5.0,
    threshold: int = 60,
    binary_output: bool = True,
    morph_close: bool = True,
    morph_kernel_size: int = 7,
    return_pil: bool = True
):
    """
    Smooth a 0-255 binary mask (uint8), soften edges to make blob-like shape.

    Args:
        mask (np.ndarray): Input mask with values 0 or 255, shape (H, W)
        blur_kernel_size (int): Size of Gaussian blur kernel
        blur_sigma (float): Sigma value for Gaussian blur
        threshold (int): Threshold value for binary output (0-255)
        binary_output (bool): If True, return 0/255 mask; if False, return soft mask
        morph_close (bool): Apply morphological closing to remove holes
        morph_kernel_size (int): Kernel size for morphological ops
        return_pil (bool): If True, return PIL image, else NumPy array

    Returns:
        PIL.Image or np.ndarray: Smoothed mask image (L mode)
    """
    assert mask.ndim == 2, "Mask must be 2D"
    assert np.issubdtype(mask.dtype, np.uint8), "Mask must be uint8"
    assert set(np.unique(mask)).issubset({0, 255}), "Mask must contain only 0 or 255"

    # Step 1: normalize to [0, 1]
    mask_norm = (mask / 255.0).astype(np.float32)

    # Step 2: blur
    blurred = cv2.GaussianBlur(mask_norm, (blur_kernel_size, blur_kernel_size), sigmaX=blur_sigma)

    # Step 3: threshold or scale
    if binary_output:
        mask_out = (blurred > (threshold / 255.0)).astype(np.uint8) * 255
    else:
        mask_out = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)

    # Step 4: morphological closing (optional)
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(mask_out).convert("L") if return_pil else mask_out
    
class FluxInpainter:
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the FLUX ControlNet inpainting model and pipeline"""
        # Load ControlNet
        controlnet = FluxControlNetModel.from_pretrained(
            paths.FLUX_INPAINT_CONTROLNET, 
            torch_dtype=self.torch_dtype
        )
        
        # Load Transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            paths.FLUX_DIR, 
            subfolder='transformer', 
            torch_dtype=self.torch_dtype
        )
        
        # Build pipeline
        self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            paths.FLUX_DIR,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=self.torch_dtype
        )
        # Resident on the GPU, FLUX.1-dev needs ~36 GB: the 12B transformer in
        # bf16 is ~24 GB on its own, plus ControlNet, T5-XXL and the VAE. On a
        # shared 40 GB card that OOMs the moment anyone else holds a few GB.
        # Model-level CPU offload keeps only the component in use resident, so
        # peak is the largest single module, at some cost in latency. Opt-in so
        # existing callers keep their current behaviour.
        if os.getenv("FLUX_CPU_OFFLOAD", "0") == "1":
            print("[FluxInpainter] model CPU offload enabled")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        
        # Ensure components are in correct dtype
        self.pipe.transformer.to(self.torch_dtype)
        self.pipe.controlnet.to(self.torch_dtype)
        
        print("Model loaded successfully")
        
    def __call__(self, image, mask, prompt="", size=(512, 512), 
                      num_inference_steps=24, controlnet_conditioning_scale=0.9,
                      guidance_scale=3.5, negative_prompt="", true_guidance_scale=3.5,
                      seed=42):
        """Run inpainting with the given parameters"""
        if self.pipe is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        mask = dilate_binary_mask(mask, size=size, kernel_size=50, iterations=1)
        mask = smooth_segmentation_mask_255(mask, blur_kernel_size=51, blur_sigma=5.0, threshold=60, binary_output=True, morph_close=True, morph_kernel_size=7, return_pil=True)

        image_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        orig_h, orig_w = image_np.shape[:2]
        mask_np = np.array(mask)
        mask_3c = np.repeat(mask_np[:, :, None] == 0, 3, axis=2)
        image_np = cv2.resize(image_np, (mask_3c.shape[1], mask_3c.shape[0]))
        masked_image = np.where(mask_3c, image_np, 255)

        masked_image_pil = Image.fromarray(masked_image)
        # masked_image_pil.save('debug/masked_image.png')

        # Run inpainting
        result = self.pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=masked_image_pil,
            control_mask=mask,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            true_guidance_scale=true_guidance_scale
        ).images[0]
        return result
        # result_np = np.array(result)
        # result_restored_np = cv2.resize(result_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # return Image.fromarray(result_restored_np)