import matplotlib.pyplot as plt
import numpy as np
import cv2


# Change the below string to the location of your data's current directory path.
#CURRENT_DIRECTORY_PATH = ""

def crop_image(img):
    return img[60:135, : ] 

def process_image_from_path(img_path):
    """
    Returns a cropped image read from the image path, img_path.
    """
    
    return (plt.imread(img_path))[60:135, : ]
    
def cropped_image_and_steer_bias(data, value):
    """ Returns a cropped version of an image and a modified 
        steer angle with a bias.
    """ 
    left_steer_angle_bias = 0.25
    right_steer_angle_bias = -0.25
    center_steer_angle_bias = 0.00

    random = np.random.randint(4)

    if random == 0:
        img = process_image_from_path(data['left'][value].strip())
        angle = float(data['steer'][value]) + left_steer_angle_bias
    elif random == 3:
        img = process_image_from_path(data['right'][value].strip())
        angle = float(data['steer'][value]) + right_steer_angle_bias
    else:
        img = process_image_from_path(data['center'][value].strip())
        angle = float(data['steer'][value]) + center_steer_angle_bias

    return img, angle


def flip_image(img, steer):
    """On average, returns the flipped version of the image, that is,
       if the image is of the car turning right, then the result is
       an image of the car turning left with all image artifacts flipped
       about the center Y-axis (This helps create more data with less training)
    """
    random = np.random.randint(1)
    if random:
        img, steer = np.fliplr(image), -steer
    return img, steer




def process_image(data, value):

    # Preprocess image by cropping it and giving it a steer bias
    img, steer = cropped_image_and_steer_bias(data, value)
    img        = img.reshape(img.shape[0], img.shape[1], 3) 
    # Randomly flip translated image and steer
    img, steer = flip_image(img, steer)
    return img, steer


def trainer(data, batch_size):
    """
    
    """
    while 1:
        batch    = data.sample(n=batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels   = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            features[i], labels[i] = process_image(data, value)
            yield np.array(features), np.array(labels)
        
def get_validation_data(data):
    """
        
    """
    while 1:
        # Skip the header row of the csv file
        for i in range(len(data)):
            img_path = data['center'][i].strip()
            img = process_image_from_path(img_path)
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steer'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang
