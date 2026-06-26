"""Policy transforms for the ``g1_mani_only_head_cam_only`` embodiment.

This embodiment is a Unitree G1 doing **manipulation only** (lower body + torso
locked on the real robot), observed through the **head camera only** (the
rendered ``ego_view``, 640x360).

The control/observation space is **16-DOF**:

    [ left_arm(7), left_gripper, right_arm(7), right_gripper ]

The arm joints are the 7 actuated arm DOF per side (shoulder pitch/roll/yaw,
elbow, wrist roll/pitch/yaw). The gripper is the scalar grip command recorded as
``left_gripper`` / ``right_gripper`` (0=open, 1=closed).

The source LeRobot dataset (built by ``wigs2universal.sh``) stores these inside
the universal joint-space heads:

    observation.state_uni_20 = [L_arm0..6, R_arm0..6, rpy(3), L_grip, R_grip, height]
    action.uni_23            = [ ...same 20..., torso_vx, torso_vy, torso_vyaw]

so the 16-DOF vector is gathered with indices [0..6, 17, 7..13, 18].
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Indices into the universal uni_20 state / uni_23 action vectors that select the
# 16-DOF manipulation-only space: L_arm(7), L_grip, R_arm(7), R_grip.
G1_MANI_16_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 17, 7, 8, 9, 10, 11, 12, 13, 18)
G1_MANI_DOF: int = len(G1_MANI_16_INDICES)  # 16


def make_g1_mani_example() -> dict:
    """Creates a random input example for the g1 manipulation policy."""
    return {
        "observation/state": np.random.rand(G1_MANI_DOF),
        "observation/image": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "prompt": "pick up the bottle",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1ManiInputs(transforms.DataTransformFn):
    """Convert g1-manipulation dataset/inference inputs into the model format.

    Used for both training and inference. Only the head camera is provided, so
    the two wrist views are zero-padded (and masked off for flow-matching pi0/pi05).
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    # If True, gather the 16-DOF manipulation subset out of the universal
    # uni_20 state / uni_23 action vectors (training path). At inference the
    # caller already provides a 16-DOF state, so leave this False there.
    gather_from_universal: bool = True

    def _gather16(self, vec) -> np.ndarray:
        vec = np.asarray(vec)
        if self.gather_from_universal and vec.shape[-1] > G1_MANI_DOF:
            return vec[..., list(G1_MANI_16_INDICES)]
        return vec

    def __call__(self, data: dict) -> dict:
        # Head camera image -> uint8 (H,W,C). LeRobot serves video frames as
        # float32 (C,H,W); inference passes uint8 (H,W,C). _parse_image handles both.
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": self._gather16(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                # No wrist cameras on this embodiment: zero-pad and mask off.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Masked off for flow-matching models; pi0-FAST does not mask padding.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = self._gather16(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1ManiOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the 16-DOF embodiment action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Actions were padded to the model action dim; keep only the first 16.
        return {"actions": np.asarray(data["actions"][..., :G1_MANI_DOF])}
