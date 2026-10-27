# FLUX ControlNet inpainting (vendored)

Inference code for the
[alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)
ControlNet, taken from
[alimama-creative/FLUX-Controlnet-Inpainting](https://github.com/alimama-creative/FLUX-Controlnet-Inpainting).
`inpainter.py` is a thin wrapper added for Fluid-Gen-Zero: it removes the
object from the first frame to give a clean static background for the warped
reference video. Weights are loaded from `FGZ_FLUX_DIR` and
`FGZ_FLUX_INPAINT_CONTROLNET` (see `fluidgen/paths.py`).
