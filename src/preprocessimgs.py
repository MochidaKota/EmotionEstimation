import mediapipe as mp
import numpy as np
import cv2
import os
from PIL import Image

fd_model_path = '/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/blaze_face_short_range.tflite'
fl_model_path = '/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/params/face_landmarker.task'

fd_model_path = os.path.abspath(fd_model_path)
fl_model_path = os.path.abspath(fl_model_path)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

fd_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=fd_model_path),
    running_mode=VisionRunningMode.IMAGE)

fl_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=fl_model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE)

facedetector = FaceDetector.create_from_options(fd_options)

facelandmarker = FaceLandmarker.create_from_options(fl_options)

def read_video(file_path):
    images = []
    video = cv2.VideoCapture(file_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    for i in range(frame_count):
        ret, frame = video.read()
        if ret:
            images.append(frame)

    return images, frame_rate, size

class PreprocessByMesh():
    def rough_trim(self, rgb_img):
        margin_rate = 0.25
        img_width = rgb_img.shape[1]
        img_height = rgb_img.shape[0]
        margin = int(img_height * margin_rate)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = facedetector.detect(mp_img)
        if not result:
            return np.ones((256, 256, 3), np.uint8)
        
        detection = result.detections[0]
        bbox = detection.bounding_box
        
        xmin = int(bbox.origin_x) - margin if int(bbox.origin_x) - margin >= 0 else 0
        ymin = int(bbox.origin_y) - margin if int(bbox.origin_y) - margin >= 0 else 0
        width = int(bbox.width) + margin * 2 if int(bbox.width) + margin * 2 <= img_width else img_width
        height = int(bbox.height) + margin * 2 if int(bbox.height) + margin * 2 <= img_height else img_height
        trim_img = rgb_img[ymin:ymin+height, xmin:xmin+width]
    
        return trim_img
    
    def rotate(self, rgb_img):
        img_width = rgb_img.shape[1]
        img_height = rgb_img.shape[0]

        rgb_img = np.array(rgb_img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = facelandmarker.detect(mp_img)
        if not result:
            return np.ones((256, 256, 3), np.uint8)

        landmarks = result.face_landmarks[0]

        top = np.array([landmarks[10].x, landmarks[10].y])
        if top[1] < 0:
            top[1] = 0
        bottom = np.array([landmarks[152].x, landmarks[152].y])
        if bottom[1] > img_height:
            bottom[1] = img_height
        face_center = (top + bottom) / 2
        vartical_vec = np.array([face_center[0], face_center[1] + 0.1])

        #culc theta
        vec1 = face_center - top
        vec2 = vartical_vec - face_center
        absvec1 = np.linalg.norm(vec1)
        absvec2 = np.linalg.norm(vec2)
        inner = np.inner(vec1, vec2)
        cos = inner/(absvec1*absvec2)
        theta = np.rad2deg(np.arccos(cos))
        if top[0] < face_center[0]:
            theta = -theta

        #rotate image
        pil_img = Image.fromarray(rgb_img)
        center = (int(face_center[0] * img_width), int(face_center[1] * img_height))
        rotated_img = pil_img.rotate(theta, center=center)
        rotated_img = np.array(rotated_img)

        return rotated_img
    
    def trim(self, rgb_img):
        img_width = rgb_img.shape[1]
        img_height = rgb_img.shape[0]
        
        rgb_img = np.array(rgb_img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = facelandmarker.detect(mp_img)
        if not result:
            return np.ones((256, 256, 3), np.uint8)
        
        landmarks = result.face_landmarks[0]
        
        top = np.array([landmarks[10].x, landmarks[10].y])
        if top[1] < 0:
            top[1] = 0
        bottom = np.array([landmarks[152].x, landmarks[152].y])
        if bottom[1] > img_height:
            bottom[1] = img_height
        face_center_x = (top[0] + bottom[0]) / 2
        face_height = int((bottom[1] - top[1]) * img_height)
        ymin = int(top[1] * img_height) if int(top[1] * img_height) >= 0 else 0
        ymax = int(bottom[1] * img_height) if int(bottom[1] * img_height) <= img_height else img_height
        xmin = int(face_center_x * img_width - face_height / 2) if int(face_center_x * img_width - face_height / 2) >= 0 else 0
        xmax = xmin + face_height if xmin + face_height <= img_width else img_width
        trim_img = rgb_img[ymin:ymax, xmin:xmax]
        
        return trim_img
    
    def __call__(self, cv2_img, size=256):
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = self.rough_trim(rgb_img)
        img = self.rotate(img)
        img = self.trim(img)
        img = cv2.resize(img, (size, size))
        
        return img  