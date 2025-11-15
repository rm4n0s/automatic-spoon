import cv2
import mediapipe as mp
import numpy as np  # For blank image creation
from controlnet_aux import MidasDetector, OpenposeDetector
from diffusers.utils import load_image
from PIL import Image
from pytsterrors import TSTError

from src.api.v1.engines.schemas import EngineSchema
from src.api.v1.jobs.schemas import ImageSchema
from src.core.enums import ControlNetType


def poses_from_reference_image(
    engine: EngineSchema, reference_image_path: str, scale: float
):
    reference_pose_image = load_image(reference_image_path)
    controlnet_conditioning_scale = []
    conditioning_images = []
    for pose_model in engine.control_net_models:
        if pose_model.control_net_type is None:
            raise TSTError(
                "no-contron-net-type-in-model",
                f"control net type of aimodel with ID {pose_model.id} is empty",
            )

        if pose_model.control_net_type == ControlNetType.OPENPOSE:
            openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            pose_image = openpose(
                reference_pose_image, include_hand=True, include_face=True
            )
            conditioning_images.append(pose_image)
            controlnet_conditioning_scale.append(scale)
        if pose_model.control_net_type == ControlNetType.MIDAS:
            midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            depth_image = midas(reference_pose_image)
            conditioning_images.append(depth_image)
            controlnet_conditioning_scale.append(scale)
        if pose_model.control_net_type == ControlNetType.MEDIAPIPE:
            pose_image = get_mediapipe_pose(reference_image_path)
            conditioning_images.append(pose_image)
            controlnet_conditioning_scale.append(scale)

    return conditioning_images, controlnet_conditioning_scale


def prepare_pose_images(engine: EngineSchema, img_sch: ImageSchema):
    if len(img_sch.control_images) == 0:
        return (None, None)

    controlnet_conditioning_scales = []
    conditioning_images = []
    for ci in img_sch.control_images:
        if ci.aimodel is None:
            cond_images, cond_scales = poses_from_reference_image(
                engine, ci.image_file_path, ci.controlnet_conditioning_scale
            )
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
