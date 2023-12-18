import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from torch.autograd import Variable


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))
        
class NoseCenteredCrop(object):
    def __init__(self, size, nose_x, nose_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.nose_x = nose_x
        self.nose_y = nose_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.nose_x - tw/2, self.nose_y - th/2, self.nose_x + tw/2, self.nose_y + th/2))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class land_transform(object):
    def __init__(self, img_size, flip_reflect):
        self.img_size = img_size
        self.flip_reflect = flip_reflect.astype(int) - 1

    def __call__(self, land, flip, offset_x, offset_y):
        land[0:len(land):2] = land[0:len(land):2] - offset_x
        land[1:len(land):2] = land[1:len(land):2] - offset_y
        # change the landmark orders when flipping
        if flip:
            land[0:len(land):2] = self.img_size - 1 - land[0:len(land):2]
            land[0:len(land):2] = land[0:len(land):2][self.flip_reflect]
            land[1:len(land):2] = land[1:len(land):2][self.flip_reflect]

        return land


class ImageTransform(object):
    def __init__(self, crop_size, phase='train'):
        self.crop_size = crop_size
        self.phase = phase
    
    def __call__(self, img, center_x, center_y):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if self.phase == 'train':
            transform = transforms.Compose([
                NoseCenteredCrop(self.crop_size, center_x, center_y),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                NoseCenteredCrop(self.crop_size, center_x, center_y),
                transforms.ToTensor(),
                normalize
            ])
        
        img = transform(img)
        return img
    
class ImageTransformV2(object):
    def __init__(self, crop_size, phase='train'):
        self.crop_size = crop_size
        self.phase = phase
    
    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if self.phase == 'train':
            transform = transforms.Compose([
                transforms.Resize(self.crop_size),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.crop_size),
                transforms.ToTensor(),
                normalize
            ])
        
        img = transform(img)
        return img
    
class GazeImageTransform(object):
    def __init__(self, detector, phase='train'):
        self.phase = phase
        self.detector = detector
        
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
        if faces is not None:
            for box, landmarks, score in faces:
                if score < .95:
                    face_img = img
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
        
        else:
            face_img = img
                
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        face_img = transform(face_img)
        face_img  = Variable(face_img)
        
        return face_img