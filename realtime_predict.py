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
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    embedding_result = embedder.embed(mp_image)
    # print(embedding_result)
    embedding_list = embedding_result.embeddings[0].embedding.tolist()
    return embedding_list
import pickle
with open("./svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
def process_image(frame):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(image,frame_timestamp_ms)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(frame, start_point, end_point, ( 0, 255, 0), 2)
        crop_img = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        send_img = image_resize(crop_img, width=480, height=480)
        
        crop_img = crop_img.astype(np.uint8)
        # cv2.imshow("cropped", crop_img)
        
        face_data=get_embadding(send_img) 
        face_data=get_embadding(crop_img) 
        
        ##########
        person_id = svm_model.predict([face_data])
        print(person_id)
        cv2.putText(frame, str(person_id[0]), (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame




video_capture = cv2.VideoCapture("./test.mp4")
# video_capture = cv2.VideoCapture(0)
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
frame_index = 0
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
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