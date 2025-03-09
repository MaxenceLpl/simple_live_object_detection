import cv2
import time
import matplotlib.pyplot as plt
import mediapipe as mp

from src.camera_and_model_initialisation import open_camera, initialise_model
from src.object_detection import visualize

def test_camera():
    try:
        cap = open_camera()
        if not cap.isOpened():
            print("Erreur : Impossible d'accéder à la caméra")
        else:
            ret, frame = cap.read()
            cap.release()  # Libération de la caméra

            if ret:
                # Conversion de l'image de BGR à RGB pour l'affichage avec matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(8, 6))
                plt.imshow(frame_rgb)
                plt.axis("off")
                plt.title("Frame")
                plt.show()
            else:
                print("Erreur : Impossible de lire une frame")
    except Exception as e:
        print("Erreur lors de l'utilisation de la caméra : ", e)

def test_flux_video(camera_index = 0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la caméra.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = int(cap.get(cv2.CAP_PROP_FPS))  # FPS théorique de la caméra
    
    print(f"Caméra détectée | Résolution : {width}x{height} | FPS théorique : {fps_camera}")

    frame_count = 0
    start_time = time.time()

    print("Démarrage du flux vidéo.")
    try:
        while True:
            frame_count += 1
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Erreur : Impossible de capturer une image.")
                break

            elapsed_time = time.time() - frame_start
            fps_real_time = 1 / elapsed_time if elapsed_time > 0 else 0

            cv2.putText(frame, f"FPS: {fps_real_time:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Appuyer sur 'q' pour quitter", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Flux Vidéo - Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Arrêt du flux vidéo.")
                break

    except KeyboardInterrupt:
        print("Flux interrompu par l'utilisateur.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Résolution : {width}x{height}")
        print(f"FPS théorique de la caméra : {fps_camera}")
        print(f"Nombre total de frames : {frame_count}")
        print(f"Durée totale : {total_time:.2f} sec")
        print(f"FPS moyen : {avg_fps:.2f}")

def test_model_sur_image(image_name = "cat_and_dog.jpg", model_name="efficientdet_lite2_float32.tflite"):
    """ Teste le modèle sur une seule image et affiche les résultats """
    # Initialiser le modèle en mode image
    image_path = f"images/{image_name}"
    detector = initialise_model(model_name, live_stream=False)

    image = mp.Image.create_from_file(image_path)

    # Effectuer la détection
    detection_result = detector.detect(image).detections

    frame = cv2.imread(image_path)  # Charger l'image pour l'affichage
    annotated_image = visualize(frame, detection_results=detection_result) 

    cv2.imshow("Détection sur Image", annotated_image)
    cv2.waitKey(0)  # Attendre une touche pour fermer
    cv2.destroyAllWindows()

def test_model_sur_une_frame(model_name="efficientdet_lite2_float32.tflite"):
    """ Teste le modèle sur une seule frame de la webcam et affiche les résultats """

    cap = open_camera()

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la caméra.")
        return

    # Initialiser le modèle en mode IMAGE (pas LIVE_STREAM)
    detector = initialise_model(model_name, live_stream=False)

    success, frame = cap.read()
    if not success:
        print("Erreur de capture de la frame.")
        cap.release()
        return

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(mp_image).detections

    annotated_image = visualize(frame, detection_results=detection_result, fps=0)  # FPS = 0 car une seule frame

    cv2.imshow("Détection sur une frame", annotated_image)
    cv2.waitKey(0)  # Attendre une touche pour fermer
    cv2.destroyAllWindows()

    # Fermer la webcam
    cap.release()



