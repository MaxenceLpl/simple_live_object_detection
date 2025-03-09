# **Simple Object Detection Project using OpenCV and MediaPipe**

This project implements **object detection** using OpenCV and MediaPipe, leveraging the **prebuilt EfficientDet Lite model**.

---

### **About EfficientDet Lite Models**
The **EfficientDet Lite** models are optimized versions of Google's **EfficientDet** object detection architecture.  
They are designed to be **lightweight and fast**, making them ideal for use on **mobile devices and embedded systems**.  
These models utilize **EfficientNet-Lite as a backbone** for feature extraction and **BiFPN (Bidirectional Feature Pyramid Network)** for efficient multi-scale object detection.  
Thanks to their **optimized structure**, they offer a **good balance between accuracy and speed**, even on low-power hardware.

---


## **Project Structure**


### **`/src` - Source Code**
This folder contains the core Python scripts.

#### **1️`camera_and_model_initialisation.py`**
Contains two functions for setting up the camera and model:
- **`open_camera()`** → Returns an **OpenCV webcam object** for the selected camera.
- **`initialise_model()`** → Returns a **MediaPipe object detector** initialized with the desired model and parameters (**explained in detail in the notebook**).

#### **2`test.py`**
Includes four testing functions:
- **`test_camera()`** → Opens the webcam and captures a **single frame**.
- **`test_flux_video()`** → Opens the webcam and displays a **live video feed with FPS information**.
- **`test_model_sur_image()`** → Initializes the selected model and applies it to an **image from the `/images` folder**.
- **`test_model_sur_frame()`** → Opens the webcam, initializes the selected model, and **detects objects in a single webcam frame**.

#### **3️`object_detection.py`**
Defines **three global variables** crucial for handling asynchronous object detection:
- `detection_results`
- `detection_times`
- `next_frame`

Functions in this file:
- **`visualize(image, fps)`** → Draws **bounding boxes, labels, and detection scores** on the image. Also displays the **FPS counter** on the frame.
- **`detection_callback(result, image, timestamp)`** → Required callback function when using **live stream mode** in MediaPipe.  
  - **Stores detected objects** in `detection_results`.  
  - **Prints frame processing time**.  
  - **Sets `next_frame = True`**, allowing the system to process the next frame.  
- **`real_time_object_detection(camera_parameters, model_parameters)`** →  
  - Opens the camera and **detects objects in real-time** using the selected model.  
  - Designed to **process only one frame at a time** to ensure synchronization between object detection, frame rendering, and webcam capture.  
  - While processing a frame, the **global variable `next_frame` is set to `False`**, preventing new frames from being processed.  
  - Once detection is complete, the **callback function sets `next_frame = True`**, allowing the next frame to be analyzed.  
  - **FPS is directly influenced by system performance** (more powerful hardware results in higher FPS).  

---

## **Notes**
- To **test different models**, download them and place them in the `/models` folder.
- Parameters can be adjusted in the **notebook** for fine-tuning the object detection performance.

---
