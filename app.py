import mediapipe as mp
import cv2
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)
detector = FaceDetector.create_from_options(options)


video_capture = cv2.VideoCapture(0)
frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
frame_index = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # detection_result = detector.detect_for_video(image, frame_timestamp_ms)
    frame_timestamp_ms = int(frame_index * (1000 / frame_rate))
    detection_result = detector.detect_for_video(image,frame_timestamp_ms)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(frame, start_point, end_point, ( 0, 255, 0), 2)
        crop_img = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        cv2.imshow("cropped", crop_img)
        

    cv2.imshow("Video", frame)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()