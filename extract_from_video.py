import mediapipe as mp
import cv2
import numpy as np
import time
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)
detector = FaceDetector.create_from_options(options)

########
import cv2 
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode


BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

new_options = ImageEmbedderOptions(
    base_options=BaseOptions(model_asset_path='mobilenet_v3_large.tflite'),
    l2_normalize=True,
    quantize=True,
    running_mode=VisionRunningMode.IMAGE)

embedder=ImageEmbedder.create_from_options(new_options)
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def get_embadding(img):
    print(img.shape)
    cv2.imshow("cropped", img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    embedding_result = embedder.embed(mp_image)
    print(embedding_result)
    embedding_list = embedding_result.embeddings[0].embedding.tolist()
    return embedding_list
# import pickle
# with open("./svm_model.pkl", "rb") as f:
#     svm_model = pickle.load(f)

import csv
import os
def save_face_data_to_csv(face_data, ids, filename):
    if not isinstance(face_data, list):
        print("Error: face_data must be a list.")
        return
    file_exists = os.path.exists(filename)
    with open(filename, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["face_data", "id"])
        
        writer.writerow([face_data, ids])
    if file_exists:
        pass 
        # print(f"Face data appended to {filename} successfully.")
    else:
        pass
        # print(f"Face data saved to {filename} successfully.")
    
def process_image(frame):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(image,frame_timestamp_ms)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x),int( bbox.origin_y)
        x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        x1, y1 = x1-19,y1-35
        x2, y2 = x2+19, y2+25
        
        crop_img = frame[y1:y2, x1:x2]
        send_img = image_resize(crop_img, width=480, height=480)
        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        # crop_img = crop_img.astype(np.uint8)
        
        ##########
        face_data=get_embadding(send_img) 
        save_face_data_to_csv(face_data, 1, "video.csv")
        # print(face_data)
        # person_id = svm_model.predict([face_data])
        # print(person_id)
        # cv2.putText(frame, str(person_id[0]), (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle (frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame




video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("test.mp4")
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
frame_index = 0
while True:
    ret, frame = video_capture.read()
    # frame = cv2.flip(frame, 1)
    if not ret:
        break
    
    # detection_result = detector.detect_for_video(image, frame_timestamp_ms)
    frame_timestamp_ms = int(frame_index * (1000 / frame_rate))
    # frame_timestamp_ms = int(time.time() * 1000)
    try:
        frame = process_image(frame)
    except:
        pass       

    cv2.imshow("Video", frame)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()