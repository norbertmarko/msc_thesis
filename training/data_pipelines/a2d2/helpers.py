import numpy as np

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image


def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                  'label_' + \
                  file_name_semantic_label[2] + '_' + \
                  file_name_semantic_label[3] + '.png'
    
    return file_name_semantic_label


# objects without classes become 0??
# ignore label as last one? (55)

class ConvertToIntLabels(object):
    def __init__(self):

        self.classes = {
            (255, 0, 0): 0,
            (200, 0, 0): 1,
            (150, 0, 0): 2,
            (128, 0, 0): 3,
            (182, 89, 6): 4,
            (150, 50, 4): 5,
            (90, 30, 1): 6,
            (90, 30, 30) : 7,
            (204, 153, 255): 8,
            (189, 73, 155): 9,
            (239, 89, 191): 10,
            (255, 128, 0) : 11,
            (200, 128, 0): 12,
            (150, 128, 0) : 13,
            (0, 255, 0): 14,
            (0, 200, 0): 15,
            (0, 150, 0) : 16,
            (0, 128, 255) : 17,
            (30, 28, 158) : 18,
            (60, 28, 100) : 19,
            (0, 255, 255): 20,
            (30, 220, 220) : 21,
            (60, 157, 199): 22,
            (255, 255, 0) : 23,
            (255, 255, 200) : 24,
            (233, 100, 0) : 25,
            (110, 110, 0) : 26,
            (128, 128, 0) : 27,
            (255, 193, 37) : 28,
            (64, 0, 64) : 29,
            (185, 122, 87): 30,
            (0, 0, 100) : 31,
            (139, 99, 108) : 32,
            (210, 50, 115) : 33,
            (255, 0, 128) : 34,
            (255, 246, 143) : 35,
            (150, 0, 150) : 36,
            (204, 255, 153) : 37,
            (238, 162, 173) : 38,
            (33, 44, 177) : 39,
            (180, 50, 180) : 40,
            (255, 70, 185) : 41,
            (238, 233, 191): 42,
            (147, 253, 194) : 43,
            (150, 150, 200) : 44,
            (180, 150, 200) : 45,
            (72, 209, 204): 46,
            (200, 125, 210) : 47,
            (159, 121, 238) : 48,
            (128, 0, 255) : 49,
            (255, 0, 255) : 50,
            (135, 206, 255) : 51,
            (241, 230, 255) : 52,
            (96, 69, 143) : 53,
            (53, 46, 82) : 54,
        }

    def __call__(self, image):
        int_array = np.zeros(shape=(256, 512), dtype=int)
        image = np.asarray(image)

        # rgb to integer
        for rgb, idx in self.classes.items():
            int_array[(image==rgb).all(2)] = idx
        return int_array

audi_cmap = [[255, 0, 0],
[200, 0, 0],
[150, 0, 0],
[128, 0, 0],
[182, 89, 6],
[150, 50, 4],
[90, 30, 1],
[90, 30, 30],
[204, 153, 255],
[189, 73, 155],
[239, 89, 191],
[255, 128, 0],
[200, 128, 0],
[150, 128, 0],
[0, 255, 0],
[0, 200, 0],
[0, 150, 0],
[0, 128, 255],
[30, 28, 158],
[60, 28, 100],
[0, 255, 255],
[30, 220, 220],
[60, 157, 199],
[255, 255, 0],
[255, 255, 200],
[233, 100, 0],
[110, 110, 0],
[128, 128, 0],
[255, 193, 37],
[64, 0, 64],
[185, 122, 87],
[0, 0, 100],
[139, 99, 108],
[210, 50, 115],
[255, 0, 128],
[255, 246, 143],
[150, 0, 150],
[204, 255, 153],
[238, 162, 173],
[33, 44, 177],
[180, 50, 180],
[255, 70, 185],
[238, 233, 191],
[147, 253, 194],
[150, 150, 200],
[180, 150, 200],
[72, 209, 204],
[200, 125, 210],
[159, 121, 238],
[128, 0, 255],
[255, 0, 255],
[135, 206, 255],
[241, 230, 255],
[96, 69, 143],
[53, 46, 82]]