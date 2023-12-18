import torch
from torchvision import transforms
from torch.autograd import Variable
from face_detection import RetinaFace

import cv2
from PIL import Image

class JAANet_ImageTransform(object):
    def __init__(self, phase='train'):
        self.phase = phase
    
    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if self.phase == 'train':
            transform = transforms.Compose([
                transforms.Resize(176),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(176),
                transforms.ToTensor(),
                normalize
            ])
        
        img = transform(img)
        return img

class L2CSNet_ImageTransform(object):
    def __init__(self, phase='train'):
        self.phase = phase
        self.detector = RetinaFace()
        
    def __call__(self, img):
        
        if self.phase == 'train':
            transform = transforms.Compose([
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        faces = self.detector.detect(img)
        face_img = img
        if faces is not None:
            for box, landmarks, score in faces:
                if score < .95:
                    continue
                x_min=int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min=int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max=int(box[2])
                y_max=int(box[3])

                face_img = img[y_min:y_max, x_min:x_max]
                
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        face_img = transform(face_img)
        face_img  = Variable(face_img)
        
        return face_img

class sixDRepNet_ImageTransform(object):
    def __init__(self, phase='train'):
        self.phase = phase
        self.detector = RetinaFace()
        
    def __call__(self, img):
        
        if self.phase == 'train':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        faces = self.detector.detect(img)
        face_img = img
        if faces is not None:
            for box, landmarks, score in faces:
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                face_img = img[y_min:y_max, x_min:x_max]
            
        face_img = Image.fromarray(face_img)
        face_img = face_img.convert('RGB')
        face_img = transform(face_img)
        
        return face_img