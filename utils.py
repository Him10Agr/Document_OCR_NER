import os
import settings
import cv2
import numpy as np
from imutils.perspective import four_point_transform

def save_upload_image(file):
    
    filename = file.filename
    name, extension = filename.split('.')
    
    save_filename = 'upload.' + extension
    upload_img_path = settings.join_path(settings.SAVE_DIR, save_filename)
    
    file.save(upload_img_path)
    return upload_img_path


def four_points_json(arr):
    
    json_points = []
    for point in arr.tolist():
        json_points.append({ 'x' : point[0], 'y':  point[1]})

    return json_points

class DocumentScan():
    
    def __init__(self):
        pass
    
    @staticmethod
    def resizer(image, width = 500):
    
        h, w, c = image.shape
        height = int((h / w) * width)
        size = (width, height)
        image = cv2.resize(image, (width, height))
        return image, size
    
    
    def document_scanner(self, image_path):
        
        self.image = cv2.imread(image_path)
        
        img_re, self.size = self.resizer(self.image)
        
        #Saving Image for JS so that co-ordinates could be changed manually
        filename = 'resize_image.jpg'
        RESIZE_IMG_PATH = settings.join_path(settings.MEDIA_DIR, filename)
        
        cv2.imwrite(RESIZE_IMG_PATH, img_re)
        
        
        ###Image Processing
        #enhance
        try:
            img_detail = cv2.detailEnhance(img_re, sigma_s = 20, sigma_r = 0.15)
            #greyscale
            img_grey = cv2.cvtColor(img_detail, cv2.COLOR_BGR2GRAY)
            #Blur (Gaussian / Average Blur)
            img_blur = cv2.GaussianBlur(img_grey, (5,5), 0)
            #edge detection
            edge_img = cv2.Canny(img_blur, 75, 200)
            #Morphological Transform
            kernel = np.ones((5, 5), np.uint8)
            dilate = cv2.dilate(edge_img, kernel, iterations = 1)
            closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
            #contours
            contours, hire = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    four_points = np.squeeze(approx)
                    break
                
            return four_points, self.size #self.size required for JS to manually adjust points
        
        except:
            
            return None, self.size


    def caliberate_to_original_size(self, four_points):
        #fourpoints will come from JS
        
        #four points for original image
        multiplier = self.image.shape[1] / self.size[0]
        four_points_orig = (four_points * multiplier).astype(int)
        
        #Wraping Image

        wrap_image = four_point_transform(self.image, four_points_orig)
        
        #apply magic color to wrap image
        magic_color_image = self.apply_brightness_contrast(wrap_image, brightness = 30, contrast = 40)
        
        return magic_color_image        
    ###magic color
    @staticmethod
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
        
        if brightness > 255 or contrast > 255 :
            print('Incorrect value for brightness or/and contrast')
            return input_img
        if brightness != 0:
            if brightness >0 :
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
                
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            
        else:
            buf = input_img.copy()
            
        if contrast != 0 :
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * ( 1 - f )
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            
        return buf

