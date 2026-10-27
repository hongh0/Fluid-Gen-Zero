"""Fluid-Gen-Zero: training-free, physics-grounded fluid-object video generation.

    fluidgen/
      reconstruction/    single image -> simulation-ready scene.json
                         (Grounding DINO + SAM2, MoGe-2, SAM 3D Objects, waterline registration)
      simulation/        scene.json -> Genesis MPM rollout -> masks, optical flow, tracks
      warped_reference/  appearance-preserving warped reference video V^sim (FLUX inpainting)
      agent/             generation-time planning: intent parsing, temporal complexity,
                         clip planning, clip-level prompts, per-backbone clip jobs
      bridge.py          intent record -> stage arguments; stage outputs -> simulation bundle
      pipeline.py        image + prompt -> simulation bundle (stages 1-3 of the paper)
      paths.py           every checkpoint / third-party path, overridable by env vars
"""
