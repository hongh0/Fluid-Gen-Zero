# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from .utils import align_frames, resize_image_ori

class CompositionAnnotator:
    def __init__(self, cfg):
        self.process_types = ["repaint", "extension", "control"]
        self.process_map = {
            "repaint": "repaint",
            "extension": "extension",
            "control": "control",
            "inpainting": "repaint",
            "outpainting": "repaint",
            "frameref": "extension",
            "clipref": "extension",
            "depth": "control",
            "flow": "control",
            "gray": "control",
            "pose": "control",
            "scribble": "control",
            "layout": "control"
        }

    def forward(self, process_type_1, process_type_2, frames_1, frames_2, masks_1, masks_2):
        total_frames = min(len(frames_1), len(frames_2), len(masks_1), len(masks_2))
        combine_type = (self.process_map[process_type_1], self.process_map[process_type_2])
        if combine_type in [("extension", "repaint"), ("extension", "control"), ("extension", "extension")]:
            output_video = [frames_2[i] * masks_1[i] + frames_1[i] * (1 - masks_1[i]) for i in range(total_frames)]
            output_mask = [masks_1[i] * masks_2[i] * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "extension"), ("control", "extension"), ("repaint", "repaint")]:
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        elif combine_type in [("repaint", "control"), ("control", "repaint")]:
            if combine_type in [("control", "repaint")]:
                frames_1, frames_2, masks_1, masks_2 = frames_2, frames_1, masks_2, masks_1
            output_video = [frames_1[i] * (1 - masks_1[i]) + frames_2[i] * masks_1[i] for i in range(total_frames)]
            output_mask = [masks_1[i] * 255 for i in range(total_frames)]
        elif combine_type in [("control", "control")]:  # apply masks_2
            output_video = [frames_1[i] * (1 - masks_2[i]) + frames_2[i] * masks_2[i] for i in range(total_frames)]
            output_mask = [(masks_1[i] * (1 - masks_2[i]) + masks_2[i] * masks_2[i]) * 255 for i in range(total_frames)]
        else:
            raise Exception("Unknown combine type")
        return output_video, output_mask


class ReferenceAnythingAnnotator:
    def __init__(self, cfg):
        from .subject import SubjectAnnotator
        self.sbjref_ins = SubjectAnnotator(cfg['SUBJECT'] if 'SUBJECT' in cfg else cfg)
        self.key_map = {
            "image": "images",
            "mask": "masks"
        }
    def forward(self, images, mode=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        for image in images:
            ret_one_data = self.sbjref_ins.forward(image=image, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
            if isinstance(ret_one_data, dict):
                for key, val in ret_one_data.items():
                    if key in self.key_map:
                        new_key = self.key_map[key]
                    else:
                        continue
                    if new_key in ret_data:
                        ret_data[new_key].append(val)
                    else:
                        ret_data[new_key] = [val]
            else:
                if 'images' in ret_data:
                    ret_data['images'].append(ret_data)
                else:
                    ret_data['images'] = [ret_data]
        return ret_data


class AnimateAnythingAnnotator:
    def __init__(self, cfg):
        from .pose import PoseBodyFaceVideoAnnotator
        self.pose_ins = PoseBodyFaceVideoAnnotator(cfg['POSE'])
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, frames=None, images=None, mode=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        ret_pose_data = self.pose_ins.forward(frames=frames)
        ret_data.update({"frames": ret_pose_data})

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class SwapAnythingAnnotator:
    def __init__(self, cfg):
        from .inpainting import InpaintingVideoAnnotator
        self.inp_ins = InpaintingVideoAnnotator(cfg['INPAINTING'])
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])

    def forward(self, video=None, frames=None, images=None, mode=None, mask=None, bbox=None, label=None, caption=None, return_mask=None, mask_cfg=None):
        ret_data = {}
        mode = mode.split(',') if ',' in mode else [mode, mode]

        ret_inp_data = self.inp_ins.forward(video=video, frames=frames, mode=mode[0], mask=mask, bbox=bbox, label=label, caption=caption, mask_cfg=mask_cfg)
        ret_data.update(ret_inp_data)

        ret_ref_data = self.ref_ins.forward(images=images, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class ExpandAnythingAnnotator:
    def __init__(self, cfg):
        from .outpainting import OutpaintingAnnotator
        from .frameref import FrameRefExpandAnnotator
        self.ref_ins = ReferenceAnythingAnnotator(cfg['REFERENCE'])
        self.frameref_ins = FrameRefExpandAnnotator(cfg['FRAMEREF'])
        self.outpainting_ins = OutpaintingAnnotator(cfg['OUTPAINTING'])

    def forward(self, images=None, mode=None, return_mask=None, mask_cfg=None, direction=None, expand_ratio=None, expand_num=None):
        ret_data = {}
        expand_image, reference_image= images[0], images[1:]
        mode = mode.split(',') if ',' in mode else ['firstframe', mode]

        outpainting_data = self.outpainting_ins.forward(expand_image,expand_ratio=expand_ratio, direction=direction)
        outpainting_image, outpainting_mask = outpainting_data['image'], outpainting_data['mask']

        frameref_data = self.frameref_ins.forward(outpainting_image,  mode=mode[0], expand_num=expand_num)
        frames, masks = frameref_data['frames'], frameref_data['masks']
        masks[0] = outpainting_mask
        ret_data.update({"frames": frames, "masks": masks})

        ret_ref_data = self.ref_ins.forward(images=reference_image, mode=mode[1], return_mask=return_mask, mask_cfg=mask_cfg)
        ret_data.update({"images": ret_ref_data['images']})

        return ret_data


class MoveAnythingAnnotator:
    def __init__(self, cfg):
        from .layout import LayoutBboxAnnotator
        self.layout_bbox_ins = LayoutBboxAnnotator(cfg['LAYOUTBBOX'])

    def forward(self, image=None, bbox=None, label=None, expand_num=None):
        frame_size = image.shape[:2]   # [H, W]
        ret_layout_data = self.layout_bbox_ins.forward(bbox, frame_size=frame_size, num_frames=expand_num, label=label)

        out_frames = [image] + ret_layout_data
        out_mask = [np.zeros(frame_size, dtype=np.uint8)] + [np.ones(frame_size, dtype=np.uint8) * 255] * len(ret_layout_data)

        ret_data = {
            "frames": out_frames,
            "masks": out_mask
        }
        return ret_data


class FirstFrameWithControlAnnotator:
    """
    Composition annotator: anchor image(s) + pre-computed control video → VACE source format.

    Mirrors the logic of ``FrameRefExpandAnnotator`` but replaces the gray placeholder
    frames with a meaningful per-frame control signal (depth maps, optical flow, etc.).

    The control video is the **primary** input — it defines both the temporal length
    and the per-frame guidance signal.  One or two anchor images are then overlaid at
    fixed positions with mask = 0 ("given"), while the remaining positions keep the
    control signal with mask = 255 ("generate following control").

    Supported anchor modes (MODE config key)
    ----------------------------------------
    ``"firstframe"``     – frame[0] = anchor, frames[1:] = control  (default)
    ``"lastframe"``      – frames[:-1] = control, frame[-1] = anchor
    ``"firstlastframe"`` – frame[0] = anchor, frames[1:-1] = control, frame[-1] = anchor2

    Typical pipeline
    ----------------
    1. Run ``DepthV2VideoAnnotator`` or ``FlowVisAnnotator`` on the source video
       to obtain ``control_frames``.
    2. Pick a reference image (e.g. the desired first frame) as ``first_frame``.
    3. Call this annotator with ``(control_frames, first_frame=...)`` to get the
       composed ``frames`` + ``masks`` ready for VACE inference.
    """

    VALID_MODES = ("firstframe", "lastframe", "firstlastframe")

    def __init__(self, cfg, device=None):
        self.mode = cfg.get("MODE", "firstframe").lower()
        assert self.mode in self.VALID_MODES, (
            f"MODE must be one of {self.VALID_MODES}, got '{self.mode}'"
        )
        self.return_dict = cfg.get("RETURN_DICT", True)
        self.return_mask = cfg.get("RETURN_MASK", True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(frame):
        if isinstance(frame, np.ndarray):
            return frame
        try:
            import torch
            if isinstance(frame, torch.Tensor):
                return frame.detach().cpu().numpy()
        except ImportError:
            pass
        try:
            from PIL import Image as PILImage
            if isinstance(frame, PILImage.Image):
                return np.array(frame)
        except ImportError:
            pass
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, frames, image=None, image_2=None,
                mode=None, return_mask=None, return_dict=None):
        """
        Parameters
        ----------
        frames : list of array-like, shape (H, W, 3)
            Pre-computed control video (depth maps / optical flow / …).
            Its length determines the total number of output frames.
            Mapped from ``--video`` in ``vace_preproccess.py``.
        image : array-like or None
            Anchor image placed at position 0 (``"firstframe"`` /
            ``"firstlastframe"`` modes). If ``None``, ``frames[0]`` is used
            unchanged (pure-control for that slot).
            Mapped from the first path of ``--image`` in ``vace_preproccess.py``.
        image_2 : array-like or None
            Anchor image placed at the last position (``"lastframe"`` /
            ``"firstlastframe"`` modes). If ``None``, ``frames[-1]`` is used
            unchanged.
            Mapped from the second path of ``--image`` in ``vace_preproccess.py``.
        mode : str or None
            Override the instance-level anchor mode at call time.
        return_mask : bool or None
            Whether to include ``masks`` in the output dict.
        return_dict : bool or None
            If ``False``, return ``(frames, masks)`` tuple instead of a dict.

        Returns
        -------
        dict (or tuple when ``return_dict=False``) with keys
            ``frames`` – composed source video, list of (H, W, 3) uint8 arrays
            ``masks``  – per-frame generation mask, list of (H, W) uint8 arrays
                         (0 = given / anchor, 255 = generate following control)
        """
        ctrl = [self._to_numpy(f) for f in frames]
        frame_num = len(ctrl)
        h, w = ctrl[0].shape[:2]

        cur_mode = (mode or self.mode).lower()
        assert cur_mode in self.VALID_MODES, f"Unknown mode '{cur_mode}'"
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_dict = return_dict if return_dict is not None else self.return_dict

        # ---- resolve anchor frames ------------------------------------ #
        ref_first = self._to_numpy(image)   if image   is not None else ctrl[0].copy()
        ref_last  = self._to_numpy(image_2) if image_2 is not None else ctrl[-1].copy()

        # Resize control frames to match the anchor resolution when they differ.
        h_ref, w_ref = ref_first.shape[:2]
        if h_ref != h or w_ref != w:
            k = h_ref / h   # approximate scale; used only to pick interpolation method
            ctrl = [resize_image_ori(h_ref, w_ref, f, k) for f in ctrl]
            h, w = h_ref, w_ref

        # Align last-frame anchor to the (now final) spatial size if needed.

        # ---- build composed output ------------------------------------ #
        # mask 0   → anchor / given frame
        # mask 255 → generate following the control signal
        keep = np.zeros((h, w), dtype=np.uint8)
        fill = np.full((h, w), 255, dtype=np.uint8)

        if cur_mode == "firstframe":
            # [anchor | ctrl ctrl ctrl …]
            out_frames = [ref_first] + ctrl[1:]
            out_masks  = [keep] + [fill.copy() for _ in range(1, frame_num)]

        elif cur_mode == "lastframe":
            # [ctrl ctrl ctrl … | anchor]
            out_frames = ctrl[:-1] + [ref_last]
            out_masks  = [fill.copy() for _ in range(frame_num - 1)] + [keep]

        else:  # "firstlastframe"
            # [anchor | ctrl ctrl … ctrl | anchor2]
            if frame_num < 2:
                raise ValueError("firstlastframe mode requires at least 2 control frames.")
            out_frames = [ref_first] + ctrl[1:-1] + [ref_last]
            out_masks  = (
                [keep]
                + [fill.copy() for _ in range(1, frame_num - 1)]
                + [keep]
            )

        if return_dict:
            ret = {"frames": out_frames}
            if return_mask:
                ret["masks"] = out_masks
            return ret
        else:
            if return_mask:
                return out_frames, out_masks
            return out_frames