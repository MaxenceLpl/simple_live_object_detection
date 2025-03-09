import cv2
import os
import mediapipe as mp

def open_camera(camera_index = 0, frame_width = 680, frame_height = 440):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    print("Webcam détectée.")
    
    return cap
    
def initialise_model(model_name, detection_callback = None, live_stream = False, score_threshold = 0.5, max_results = 3, target_classes = None):
    model_path = f"models/{model_name}"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print("Modèle trouvé :", model_path)

    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    running_mode = VisionRunningMode.LIVE_STREAM if live_stream else VisionRunningMode.IMAGE
    
    options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=running_mode,
    score_threshold=score_threshold,
    result_callback=detection_callback if live_stream else None,
    max_results=max_results,
    category_allowlist=target_classes if target_classes else None
    )
    
    return ObjectDetector.create_from_options(options)
    