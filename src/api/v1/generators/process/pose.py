# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

import cv2
import mediapipe as mp
import numpy as np  # For blank image creation
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector
from diffusers.utils import load_image
from PIL import Image
from pytsterrors import TSTError

from src.api.v1.engines.schemas import EngineSchema
from src.api.v1.images.schemas import ControlNetImageSchema, ImageSchema
from src.core.enums import ControlNetType


def poses_from_reference_image(engine: EngineSchema, ci: ControlNetImageSchema):
    reference_pose_image = load_image(ci.image_file_path)
    controlnet_conditioning_scales = []
    conditioning_images = []
    scale = engine.controlnet_conditioning_scale
    if ci.controlnet_conditioning_scale:
        scale = ci.controlnet_conditioning_scale

    assert scale is not None
    for pose_model in engine.control_net_models:
        if pose_model.control_net_type is None:
            raise TSTError(
                "no-contron-net-type-in-model",
                f"control net type of aimodel with ID {pose_model.id} is empty",
            )

        match pose_model.control_net_type:
            case ControlNetType.OPENPOSE:
                openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                pose_image = openpose(
                    reference_pose_image, include_hand=True, include_face=True
                )
                _ = pose_image.save(ci.image_file_path + "_openpose.png")
                conditioning_images.append(pose_image)
                controlnet_conditioning_scales.append(scale)
            case ControlNetType.MIDAS:
                midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
                depth_image = midas(reference_pose_image)

                # Convert to numpy for processing
                depth_array = (
                    np.array(depth_image).astype(np.float32) / 255.0
                )  # Normalize to [0,1]

                # Optional: Invert if needed (test visually)
                # depth_array = 1.0 - depth_array

                # Enhance contrast for better hand distinction (optional, adjust params)
                depth_array = np.clip(
                    (depth_array - 0.2) / 0.6, 0, 1
                )  # Stretch contrast

                # Back to PIL for ControlNet
                processed_depth = Image.fromarray((depth_array * 255).astype(np.uint8))
                _ = processed_depth.save(ci.image_file_path + "_midas.png")  # pyright: ignore[reportAttributeAccessIssue]

                conditioning_images.append(processed_depth)
                controlnet_conditioning_scales.append(scale)

            case ControlNetType.MEDIAPIPE:
                pose_image = get_mediapipe_pose(ci.image_file_path)
                _ = pose_image.save(ci.image_file_path + "_mediapipe.png")
                conditioning_images.append(pose_image)
                controlnet_conditioning_scales.append(scale)
            case ControlNetType.CANNY:
                canny_detector = CannyDetector()
                assert ci.canny_low_threshold is not None
                assert ci.canny_high_threshold is not None
                input_image = Image.open(ci.image_file_path).convert("RGB")
                pose_image = canny_detector(
                    input_image,
                    low_threshold=ci.canny_low_threshold,
                    high_threshold=ci.canny_high_threshold,
                )
                _ = pose_image.save(ci.image_file_path + "_canny.png")

                conditioning_images.append(pose_image)
                controlnet_conditioning_scales.append(scale)

    return conditioning_images, controlnet_conditioning_scales


def prepare_pose_images(engine: EngineSchema, img_sch: ImageSchema):
    controlnet_conditioning_scales = []
    conditioning_images = []
    for ci in img_sch.control_images:
        if ci.aimodel is None:
            cond_images, cond_scales = poses_from_reference_image(engine, ci)
            conditioning_images = conditioning_images + cond_images
            controlnet_conditioning_scales = (
                controlnet_conditioning_scales + cond_scales
            )
        else:
            img = load_image(ci.image_file_path)
            conditioning_images.append(img)
            controlnet_conditioning_scales.append(ci.controlnet_conditioning_scale)

    return (conditioning_images, controlnet_conditioning_scales)


def get_mediapipe_pose(reference_image_path: str):
    # Load MediaPipe Pose
    mp_pose = mp.solutions.pose  # pyright: ignore[reportAttributeAccessIssue]
    mp_drawing = mp.solutions.drawing_utils  # pyright: ignore[reportAttributeAccessIssue]
    mp_drawing_styles = mp.solutions.drawing_styles  # pyright: ignore[reportAttributeAccessIssue]
    pose_detector = mp_pose.Pose(
        static_image_mode=True,  # For single images
        model_complexity=2,  # Higher accuracy for complex poses like crossed legs
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    # Load reference image
    reference_pose_image = load_image(reference_image_path).convert("RGB")
    reference_pose_cv = np.array(reference_pose_image)
    reference_pose_cv = cv2.cvtColor(reference_pose_cv, cv2.COLOR_RGB2BGR)

    # Detect pose
    results = pose_detector.process(cv2.cvtColor(reference_pose_cv, cv2.COLOR_BGR2RGB))

    # Create blank black canvas for skeleton
    height, width, _ = reference_pose_cv.shape
    pose_image_cv = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw skeleton if landmarks detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            pose_image_cv,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Convert to PIL Image for pipeline
    return Image.fromarray(cv2.cvtColor(pose_image_cv, cv2.COLOR_BGR2RGB))
