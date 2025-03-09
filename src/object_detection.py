import cv2
import time
import mediapipe as mp

from src.camera_and_model_initialisation import open_camera, initialise_model

detection_results = [] #Permet de stoquer les resultats de detection d'objet
detection_times = [] #Permet de stoquer le temps entre l'image et la detection d'objet sur cette image
next_frame = True #Permet d'arreter la camera tant que le model n'a pas termine de process l'image pour eviter l'accumulation de latence

#Cette fonction permet d'afficher la frame de la camera ainsi que des box autout des objets detectes et les fps
def visualize(image, detection_results = [], fps = None):
    detected_objects = []
    
    for detection in detection_results:
        class_name = detection.categories[0].category_name
        score = detection.categories[0].score

        bbox = detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        detected_objects.append(f"{class_name}: {score:.2f}")

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if fps:
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    height = image.shape[0]
    cv2.putText(image, "Appuyer sur 'q' pour quitter", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if detected_objects:
        print(f"Objets détectés : {', '.join(detected_objects)}")
    else:
        print("Aucun objet détecté.")

    return image

#Cette fonction est le callback appellee lorsque le modele a termine le traitement d'une image, et mets a jour la liste de detection
def detection_callback(result, image, timestamp):
    global detection_results, detection_times, next_frame
    detection_results = result.detections
    detection_time = time.time() - detection_times.pop(0)
    next_frame = True
    print(f"Callback reçu - Temps de traitement : {detection_time:.2f}s")
    
# Lance la detection en temps reel
def real_time_object_detection(camera_parameters, model_parameters):
    global detection_results, detection_times, next_frame
    
    model_parameters["live_stream"] = True
    model_parameters["detection_callback"] = detection_callback
    
    cap = open_camera(**camera_parameters)
    detector = initialise_model(**model_parameters)
    
    try:
        while cap.isOpened():
            start_time = time.time()

            success, frame = cap.read()
            if not success:
                print("Erreur de flux vidéo.")
                break
            
            # Convertir en format MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_times.append(time.time())

            next_frame = False # Bloque les prochaines images jusqu'a la fin de la detection d'objets par le modele
            detector.detect_async(mp_image, int(time.time() * 1000))
            
            while not next_frame:
                time.sleep(0.05)

            #Estimation des fps
            fps = 1/(time.time() - start_time)
            annotated_image = visualize(frame, detection_results, fps)
            cv2.imshow("Détection en temps réel", annotated_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Keyboard Interrupt.")
                break

    except KeyboardInterrupt:
        print("Arrêt manuel.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    
