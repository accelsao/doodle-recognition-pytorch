import ast
import sys
import numpy as np
import cv2 as cv
import torch
from PyQt5.QtCore import QMimeData, QPointF, Qt, QObject, pyqtSlot, QSize, QAbstractListModel, QRectF, QPoint, QRect, \
    QTimer, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtGui import QImage, QPixmap, QDrag, QPainter, QStandardItemModel, QIcon, QPen, QColor, QCursor
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QGridLayout,
                             QLabel, QPushButton, QWidget, QVBoxLayout, QListWidget, QAbstractItemView, QHBoxLayout,
                             QListView, QListWidgetItem, QMainWindow, QStackedWidget, QStackedLayout, QMenu, QMenuBar,
                             QAction, QSpacerItem, QSizePolicy, QSlider)

import pandas as pd
import tqdm
import glob
import os
from simplification.cutil import simplify_coords
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
import random


class DoodleDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train', nrows=1000, skiprows=None,
                 size=256, thickness=2, transform=None):
        super(DoodleDataset, self).__init__()

        self.thickness = thickness
        self.root_dir = root_dir
        self.mode = mode
        self.size = size
        self.transform = transform
        file = os.path.join(self.root_dir, csv_file)
        self.data = pd.read_csv(file, usecols=['drawing'], nrows=nrows, skiprows=skiprows)

        if self.mode == 'train':
            self.label = self.get_label(csv_file)

    @staticmethod
    def draw(raw_strokes, size=256, thickness=2, color_by_time=True):
        img = np.zeros((255, 255), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(1, len(stroke[0])):
                color = 255 - min(t, 10) * 13 if color_by_time else 255
                cv.line(img, (stroke[0][i - 1], stroke[1][i - 1]), (stroke[0][i], stroke[1][i]), color, thickness)

        if size != 255:
            img = cv.resize(img, (size, size))

        return img

    def get_label(self, csv_file):
        return window.label_dict[csv_file.split('/')[-1].replace(' ', '_')[:-4]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_strokes = ast.literal_eval(self.data.drawing[idx])
        sample = self.draw(raw_strokes, self.size, self.thickness)

        if self.transform:
            sample = self.transform(sample)
        if self.mode == 'train':
            # sample[None] equivalent to numpy.newaxis
            return (sample[None] / 255).astype('float32'), self.label
        else:
            return (sample[None] / 255).astype('float32')

# ref: https://medium.com/floatflower-me/qt%E4%B8%ADqthreadpool%E7%B7%9A%E7%A8%8B%E6%B1%A0%E5%AF%A6%E7%8F%BE%E8%88%87%E5%8E%9F%E5%A7%8B%E7%A2%BC%E8%A7%A3%E6%9E%90-5d1a67c1480b
# ref: https://stackoverflow.com/questions/47560399/run-function-in-the-background-and-update-ui
class ProcessRunnable(QRunnable):
    def __init__(self, target, args):
        QRunnable.__init__(self)
        self.t = target
        self.args = args

    def run(self):
        self.t(*self.args)

    def start(self):
        QThreadPool.globalInstance().start(self)



class DrawBoard(QLabel):
    def __init__(self):
        super(DrawBoard, self).__init__()
        self.setFixedSize(QSize(800, 600))
        print(self.size())
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        self.image.fill(Qt.white)
        self.imageDraw = QImage(self.size(), QImage.Format_ARGB32)
        self.imageDraw.fill(Qt.transparent)
        # self.imageDraw.fill(Qt.white)

        self.drawing = False
        self.brushSize = 2
        self._clear_size = 20
        self.brushColor = QColor(Qt.black)
        self.lastPoint = QPoint()
        self.change = False
        self.strokes = []
        self.stroke = []



        self.batch_size = 128
        self.epochs = 1
        self.input_nc = 64
        self.lr = 2e-3
        self.image_size = 224
        self.select_nrows = 10000
        self.num_classes = 340

        self.print_freq = 1000

        self.create_label()
        self.init_model()
        # self.compute()




    def create_label(self):
        self.label_dict = {'The_Eiffel_Tower': 0, 'The_Great_Wall_of_China': 1, 'The_Mona_Lisa': 2, 'airplane': 3,
                           'alarm_clock': 4, 'ambulance': 5, 'angel': 6, 'animal_migration': 7, 'ant': 8, 'anvil': 9,
                           'apple': 10, 'arm': 11, 'asparagus': 12, 'axe': 13, 'backpack': 14, 'banana': 15,
                           'bandage': 16,
                           'barn': 17, 'baseball_bat': 18, 'baseball': 19, 'basket': 20, 'basketball': 21, 'bat': 22,
                           'bathtub': 23, 'beach': 24, 'bear': 25, 'beard': 26, 'bed': 27, 'bee': 28, 'belt': 29,
                           'bench': 30, 'bicycle': 31, 'binoculars': 32, 'bird': 33, 'birthday_cake': 34,
                           'blackberry': 35,
                           'blueberry': 36, 'book': 37, 'boomerang': 38, 'bottlecap': 39, 'bowtie': 40, 'bracelet': 41,
                           'brain': 42, 'bread': 43, 'bridge': 44, 'broccoli': 45, 'broom': 46, 'bucket': 47,
                           'bulldozer': 48, 'bus': 49, 'bush': 50, 'butterfly': 51, 'cactus': 52, 'cake': 53,
                           'calculator': 54, 'calendar': 55, 'camel': 56, 'camera': 57, 'camouflage': 58,
                           'campfire': 59,
                           'candle': 60, 'cannon': 61, 'canoe': 62, 'car': 63, 'carrot': 64, 'castle': 65, 'cat': 66,
                           'ceiling_fan': 67, 'cell_phone': 68, 'cello': 69, 'chair': 70, 'chandelier': 71,
                           'church': 72,
                           'circle': 73, 'clarinet': 74, 'clock': 75, 'cloud': 76, 'coffee_cup': 77, 'compass': 78,
                           'computer': 79, 'cookie': 80, 'cooler': 81, 'couch': 82, 'cow': 83, 'crab': 84, 'crayon': 85,
                           'crocodile': 86, 'crown': 87, 'cruise_ship': 88, 'cup': 89, 'diamond': 90, 'dishwasher': 91,
                           'diving_board': 92, 'dog': 93, 'dolphin': 94, 'donut': 95, 'door': 96, 'dragon': 97,
                           'dresser': 98, 'drill': 99, 'drums': 100, 'duck': 101, 'dumbbell': 102, 'ear': 103,
                           'elbow': 104,
                           'elephant': 105, 'envelope': 106, 'eraser': 107, 'eye': 108, 'eyeglasses': 109, 'face': 110,
                           'fan': 111, 'feather': 112, 'fence': 113, 'finger': 114, 'fire_hydrant': 115,
                           'fireplace': 116,
                           'firetruck': 117, 'fish': 118, 'flamingo': 119, 'flashlight': 120, 'flip_flops': 121,
                           'floor_lamp': 122, 'flower': 123, 'flying_saucer': 124, 'foot': 125, 'fork': 126,
                           'frog': 127,
                           'frying_pan': 128, 'garden_hose': 129, 'garden': 130, 'giraffe': 131, 'goatee': 132,
                           'golf_club': 133, 'grapes': 134, 'grass': 135, 'guitar': 136, 'hamburger': 137,
                           'hammer': 138,
                           'hand': 139, 'harp': 140, 'hat': 141, 'headphones': 142, 'hedgehog': 143, 'helicopter': 144,
                           'helmet': 145, 'hexagon': 146, 'hockey_puck': 147, 'hockey_stick': 148, 'horse': 149,
                           'hospital': 150, 'hot_air_balloon': 151, 'hot_dog': 152, 'hot_tub': 153, 'hourglass': 154,
                           'house_plant': 155, 'house': 156, 'hurricane': 157, 'ice_cream': 158, 'jacket': 159,
                           'jail': 160,
                           'kangaroo': 161, 'key': 162, 'keyboard': 163, 'knee': 164, 'ladder': 165, 'lantern': 166,
                           'laptop': 167, 'leaf': 168, 'leg': 169, 'light_bulb': 170, 'lighthouse': 171,
                           'lightning': 172,
                           'line': 173, 'lion': 174, 'lipstick': 175, 'lobster': 176, 'lollipop': 177, 'mailbox': 178,
                           'map': 179, 'marker': 180, 'matches': 181, 'megaphone': 182, 'mermaid': 183,
                           'microphone': 184,
                           'microwave': 185, 'monkey': 186, 'moon': 187, 'mosquito': 188, 'motorbike': 189,
                           'mountain': 190,
                           'mouse': 191, 'moustache': 192, 'mouth': 193, 'mug': 194, 'mushroom': 195, 'nail': 196,
                           'necklace': 197, 'nose': 198, 'ocean': 199, 'octagon': 200, 'octopus': 201, 'onion': 202,
                           'oven': 203, 'owl': 204, 'paint_can': 205, 'paintbrush': 206, 'palm_tree': 207, 'panda': 208,
                           'pants': 209, 'paper_clip': 210, 'parachute': 211, 'parrot': 212, 'passport': 213,
                           'peanut': 214,
                           'pear': 215, 'peas': 216, 'pencil': 217, 'penguin': 218, 'piano': 219, 'pickup_truck': 220,
                           'picture_frame': 221, 'pig': 222, 'pillow': 223, 'pineapple': 224, 'pizza': 225,
                           'pliers': 226,
                           'police_car': 227, 'pond': 228, 'pool': 229, 'popsicle': 230, 'postcard': 231, 'potato': 232,
                           'power_outlet': 233, 'purse': 234, 'rabbit': 235, 'raccoon': 236, 'radio': 237, 'rain': 238,
                           'rainbow': 239, 'rake': 240, 'remote_control': 241, 'rhinoceros': 242, 'river': 243,
                           'roller_coaster': 244, 'rollerskates': 245, 'sailboat': 246, 'sandwich': 247, 'saw': 248,
                           'saxophone': 249, 'school_bus': 250, 'scissors': 251, 'scorpion': 252, 'screwdriver': 253,
                           'sea_turtle': 254, 'see_saw': 255, 'shark': 256, 'sheep': 257, 'shoe': 258, 'shorts': 259,
                           'shovel': 260, 'sink': 261, 'skateboard': 262, 'skull': 263, 'skyscraper': 264,
                           'sleeping_bag': 265, 'smiley_face': 266, 'snail': 267, 'snake': 268, 'snorkel': 269,
                           'snowflake': 270, 'snowman': 271, 'soccer_ball': 272, 'sock': 273, 'speedboat': 274,
                           'spider': 275, 'spoon': 276, 'spreadsheet': 277, 'square': 278, 'squiggle': 279,
                           'squirrel': 280,
                           'stairs': 281, 'star': 282, 'steak': 283, 'stereo': 284, 'stethoscope': 285, 'stitches': 286,
                           'stop_sign': 287, 'stove': 288, 'strawberry': 289, 'streetlight': 290, 'string_bean': 291,
                           'submarine': 292, 'suitcase': 293, 'sun': 294, 'swan': 295, 'sweater': 296, 'swing_set': 297,
                           'sword': 298, 't-shirt': 299, 'table': 300, 'teapot': 301, 'teddy-bear': 302,
                           'telephone': 303,
                           'television': 304, 'tennis_racquet': 305, 'tent': 306, 'tiger': 307, 'toaster': 308,
                           'toe': 309,
                           'toilet': 310, 'tooth': 311, 'toothbrush': 312, 'toothpaste': 313, 'tornado': 314,
                           'tractor': 315, 'traffic_light': 316, 'train': 317, 'tree': 318, 'triangle': 319,
                           'trombone': 320, 'truck': 321, 'trumpet': 322, 'umbrella': 323, 'underwear': 324, 'van': 325,
                           'vase': 326, 'violin': 327, 'washing_machine': 328, 'watermelon': 329, 'waterslide': 330,
                           'whale': 331, 'wheel': 332, 'windmill': 333, 'wine_bottle': 334, 'wine_glass': 335,
                           'wristwatch': 336, 'yoga': 337, 'zebra': 338, 'zigzag': 339}
        self.dec_dict = {0: 'The_Eiffel_Tower', 1: 'The_Great_Wall_of_China', 2: 'The_Mona_Lisa', 3: 'airplane',
                         4: 'alarm_clock', 5: 'ambulance', 6: 'angel', 7: 'animal_migration', 8: 'ant', 9: 'anvil',
                         10: 'apple', 11: 'arm', 12: 'asparagus', 13: 'axe', 14: 'backpack', 15: 'banana',
                         16: 'bandage',
                         17: 'barn', 18: 'baseball_bat', 19: 'baseball', 20: 'basket', 21: 'basketball', 22: 'bat',
                         23: 'bathtub', 24: 'beach', 25: 'bear', 26: 'beard', 27: 'bed', 28: 'bee', 29: 'belt',
                         30: 'bench',
                         31: 'bicycle', 32: 'binoculars', 33: 'bird', 34: 'birthday_cake', 35: 'blackberry',
                         36: 'blueberry', 37: 'book', 38: 'boomerang', 39: 'bottlecap', 40: 'bowtie', 41: 'bracelet',
                         42: 'brain', 43: 'bread', 44: 'bridge', 45: 'broccoli', 46: 'broom', 47: 'bucket',
                         48: 'bulldozer',
                         49: 'bus', 50: 'bush', 51: 'butterfly', 52: 'cactus', 53: 'cake', 54: 'calculator',
                         55: 'calendar',
                         56: 'camel', 57: 'camera', 58: 'camouflage', 59: 'campfire', 60: 'candle', 61: 'cannon',
                         62: 'canoe', 63: 'car', 64: 'carrot', 65: 'castle', 66: 'cat', 67: 'ceiling_fan',
                         68: 'cell_phone',
                         69: 'cello', 70: 'chair', 71: 'chandelier', 72: 'church', 73: 'circle', 74: 'clarinet',
                         75: 'clock', 76: 'cloud', 77: 'coffee_cup', 78: 'compass', 79: 'computer', 80: 'cookie',
                         81: 'cooler', 82: 'couch', 83: 'cow', 84: 'crab', 85: 'crayon', 86: 'crocodile', 87: 'crown',
                         88: 'cruise_ship', 89: 'cup', 90: 'diamond', 91: 'dishwasher', 92: 'diving_board', 93: 'dog',
                         94: 'dolphin', 95: 'donut', 96: 'door', 97: 'dragon', 98: 'dresser', 99: 'drill', 100: 'drums',
                         101: 'duck', 102: 'dumbbell', 103: 'ear', 104: 'elbow', 105: 'elephant', 106: 'envelope',
                         107: 'eraser', 108: 'eye', 109: 'eyeglasses', 110: 'face', 111: 'fan', 112: 'feather',
                         113: 'fence', 114: 'finger', 115: 'fire_hydrant', 116: 'fireplace', 117: 'firetruck',
                         118: 'fish',
                         119: 'flamingo', 120: 'flashlight', 121: 'flip_flops', 122: 'floor_lamp', 123: 'flower',
                         124: 'flying_saucer', 125: 'foot', 126: 'fork', 127: 'frog', 128: 'frying_pan',
                         129: 'garden_hose',
                         130: 'garden', 131: 'giraffe', 132: 'goatee', 133: 'golf_club', 134: 'grapes', 135: 'grass',
                         136: 'guitar', 137: 'hamburger', 138: 'hammer', 139: 'hand', 140: 'harp', 141: 'hat',
                         142: 'headphones', 143: 'hedgehog', 144: 'helicopter', 145: 'helmet', 146: 'hexagon',
                         147: 'hockey_puck', 148: 'hockey_stick', 149: 'horse', 150: 'hospital', 151: 'hot_air_balloon',
                         152: 'hot_dog', 153: 'hot_tub', 154: 'hourglass', 155: 'house_plant', 156: 'house',
                         157: 'hurricane', 158: 'ice_cream', 159: 'jacket', 160: 'jail', 161: 'kangaroo', 162: 'key',
                         163: 'keyboard', 164: 'knee', 165: 'ladder', 166: 'lantern', 167: 'laptop', 168: 'leaf',
                         169: 'leg', 170: 'light_bulb', 171: 'lighthouse', 172: 'lightning', 173: 'line', 174: 'lion',
                         175: 'lipstick', 176: 'lobster', 177: 'lollipop', 178: 'mailbox', 179: 'map', 180: 'marker',
                         181: 'matches', 182: 'megaphone', 183: 'mermaid', 184: 'microphone', 185: 'microwave',
                         186: 'monkey', 187: 'moon', 188: 'mosquito', 189: 'motorbike', 190: 'mountain', 191: 'mouse',
                         192: 'moustache', 193: 'mouth', 194: 'mug', 195: 'mushroom', 196: 'nail', 197: 'necklace',
                         198: 'nose', 199: 'ocean', 200: 'octagon', 201: 'octopus', 202: 'onion', 203: 'oven',
                         204: 'owl',
                         205: 'paint_can', 206: 'paintbrush', 207: 'palm_tree', 208: 'panda', 209: 'pants',
                         210: 'paper_clip', 211: 'parachute', 212: 'parrot', 213: 'passport', 214: 'peanut',
                         215: 'pear',
                         216: 'peas', 217: 'pencil', 218: 'penguin', 219: 'piano', 220: 'pickup_truck',
                         221: 'picture_frame', 222: 'pig', 223: 'pillow', 224: 'pineapple', 225: 'pizza', 226: 'pliers',
                         227: 'police_car', 228: 'pond', 229: 'pool', 230: 'popsicle', 231: 'postcard', 232: 'potato',
                         233: 'power_outlet', 234: 'purse', 235: 'rabbit', 236: 'raccoon', 237: 'radio', 238: 'rain',
                         239: 'rainbow', 240: 'rake', 241: 'remote_control', 242: 'rhinoceros', 243: 'river',
                         244: 'roller_coaster', 245: 'rollerskates', 246: 'sailboat', 247: 'sandwich', 248: 'saw',
                         249: 'saxophone', 250: 'school_bus', 251: 'scissors', 252: 'scorpion', 253: 'screwdriver',
                         254: 'sea_turtle', 255: 'see_saw', 256: 'shark', 257: 'sheep', 258: 'shoe', 259: 'shorts',
                         260: 'shovel', 261: 'sink', 262: 'skateboard', 263: 'skull', 264: 'skyscraper',
                         265: 'sleeping_bag', 266: 'smiley_face', 267: 'snail', 268: 'snake', 269: 'snorkel',
                         270: 'snowflake', 271: 'snowman', 272: 'soccer_ball', 273: 'sock', 274: 'speedboat',
                         275: 'spider',
                         276: 'spoon', 277: 'spreadsheet', 278: 'square', 279: 'squiggle', 280: 'squirrel',
                         281: 'stairs',
                         282: 'star', 283: 'steak', 284: 'stereo', 285: 'stethoscope', 286: 'stitches',
                         287: 'stop_sign',
                         288: 'stove', 289: 'strawberry', 290: 'streetlight', 291: 'string_bean', 292: 'submarine',
                         293: 'suitcase', 294: 'sun', 295: 'swan', 296: 'sweater', 297: 'swing_set', 298: 'sword',
                         299: 't-shirt', 300: 'table', 301: 'teapot', 302: 'teddy-bear', 303: 'telephone',
                         304: 'television', 305: 'tennis_racquet', 306: 'tent', 307: 'tiger', 308: 'toaster',
                         309: 'toe',
                         310: 'toilet', 311: 'tooth', 312: 'toothbrush', 313: 'toothpaste', 314: 'tornado',
                         315: 'tractor',
                         316: 'traffic_light', 317: 'train', 318: 'tree', 319: 'triangle', 320: 'trombone',
                         321: 'truck',
                         322: 'trumpet', 323: 'umbrella', 324: 'underwear', 325: 'van', 326: 'vase', 327: 'violin',
                         328: 'washing_machine', 329: 'watermelon', 330: 'waterslide', 331: 'whale', 332: 'wheel',
                         333: 'windmill', 334: 'wine_bottle', 335: 'wine_glass', 336: 'wristwatch', 337: 'yoga',
                         338: 'zebra', 339: 'zigzag'}



    def init_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        filename_pth = 'pretrained/checkpoint_mobilenetv2.pth'

        model = mobilenet_v2()

        def squeeze_weights(m):
            m.weight.data = m.weight.data.sum(1, keepdim=True)
            m.in_channels = 1

        model.features[0][0].apply(squeeze_weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, self.num_classes),
        )
        print(model)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 12000, 18000], gamma=0.5)
        model.load_state_dict(torch.load(filename_pth, map_location=device))
        model.eval()

        self.model = model
        self.device = device


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
        self.stroke = [[self.lastPoint.x(), self.lastPoint.y()]]

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.imageDraw)
            # painter = QPainter(self)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if self.change:
                r = QRect(QPoint(), self._clear_size * QSize())
                r.moveCenter(event.pos())
                painter.save()
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.eraseRect(r)
                painter.restore()
            else:
                painter.drawLine(self.lastPoint, event.pos())
            painter.end()
            self.lastPoint = event.pos()
            self.stroke.append([self.lastPoint.x(), self.lastPoint.y()])
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

        self.strokes.append(self.stroke)
        self.stroke = []

    def paintEvent(self, event):
        super(DrawBoard, self).paintEvent(event)

        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        canvasPainter.drawImage(self.rect(), self.imageDraw, self.imageDraw.rect())

    def clear_board(self):
        self.imageDraw = QImage(self.size(), QImage.Format_ARGB32)
        self.imageDraw.fill(Qt.transparent)
        self.update()
        self.strokes = []

    @staticmethod
    def draw(raw_strokes, size=256, thickness=2, color_by_time=True):
        img = np.zeros((255, 255), np.uint8)
        print(raw_strokes)
        for t, stroke in enumerate(raw_strokes):
            print(stroke)
            for i in range(1, len(stroke[0])):
                color = 255 - min(t, 10) * 13 if color_by_time else 255
                cv.line(img, (stroke[0][i - 1], stroke[1][i - 1]), (stroke[0][i], stroke[1][i]), color, thickness)

        if size != 255:
            img = cv.resize(img, (size, size))

        return img

    def compute(self):


        x_min, x_max = 800, 0
        y_min, y_max = 600, 0
        for stk in self.strokes:
            p = np.split(np.array(stk), [-1], axis=1)
            # print(p)
            p = np.array(p)
            # print(np.min(p[0]), np.max(p[0]))
            # print(np.min(p[1]), np.max(p[1]))
            x_min = min(x_min, np.min(p[0]))
            x_max = max(x_max, np.max(p[0]))
            y_min = min(y_min, np.min(p[1]))
            y_max = max(y_max, np.max(p[1]))


        stks = []
        for stk in self.strokes:

            p = np.split(np.array(stk), [-1], axis=1)
            p = np.array(p).astype(np.float32).squeeze(-1)

            p[0] -= x_min
            p[1] -= y_min
            p[0] *= 255.0 / (x_max - x_min)
            p[1] *= 255.0 / (y_max - y_min)

            p = p.astype(np.uint8)

            a = np.array(p[0])
            b = np.array(p[1])
            # print(a, b)
            p = np.stack((a, b), axis=-1)
            # print(p)


            p = simplify_coords(p, 2.0)
            p = np.split(p, [-1], axis=1)
            # print(p)

            p = np.array(p).squeeze(-1).astype(np.uint8)
            stks.append(p.tolist())


        data = pd.DataFrame([[9000003627287624, 'UA', stks.__str__()]], columns=['key_id', 'countrycode', 'drawing'])
        data.to_csv('dataset/test_simplified/tmp.csv')

        # img = self.draw(stks)
        # cv.imwrite('images/sim_tmp.png', img)



        # dataset/test_simplified/tmp.csv

        testset = DoodleDataset('tmp.csv', 'dataset/test_simplified', mode='test', nrows=None,
                                size=self.image_size)
        testloader = DataLoader(testset, batch_size=1)
        # #
        labels = np.empty((0, 3))
        for img in tqdm.tqdm(testloader):
            img = img.to(self.device)
            output = self.model(img)
            _, pred = output.topk(3, 1, True, True)
            labels = np.concatenate([labels, pred.cpu()], axis=0)

        top3 = []
        for i, label in enumerate(labels):
            for l in label:
                top3.append(self.dec_dict[l])
        # #
        print(top3)

        if window.label in top3:
            window.endGame(1)

        window.update_top3(top3)
        # print('compute end : {}'.format(window.timeLeft))

        # return top3
        # return ['a', 'b', 'c']








class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        top, left, width, height = 400, 400, 800, 600
        self.setWindowTitle("MyPainter")
        # self.setGeometry(top, left, width, height)

        # self.image = QImage(self.size(), QImage.Format_ARGB32)
        # self.image.fill(Qt.white)
        # self.imageDraw = QImage(self.size(), QImage.Format_ARGB32)
        # self.imageDraw.fill(Qt.transparent)

        # self.drawing = False
        # self.brushSize = 2
        # self._clear_size = 20
        # self.brushColor = QColor(Qt.black)
        # self.lastPoint = QPoint()


        # self.drawboard.resize(QSize(800, 600))
        # self.resize(self.drawboard.sizeHint())
        # self.drawboard.setFixedSize(QSize(800, 600))
        self.setFixedSize(QSize(800, 600))
        self.drawboard = DrawBoard()



        guess_label = QLabel()
        time_counter = QLabel()
        top3_label = QLabel()

        viewer = QWidget()
        hbox = QHBoxLayout()
        viewer.setLayout(hbox)
        hbox.addWidget(guess_label)
        hbox.addWidget(time_counter)
        hbox.addWidget(top3_label)

        self.time_counter = time_counter
        self.guess_label = guess_label
        self.top3_label = top3_label
        self.viewer = viewer

        mainMenu = self.menuBar()
        # mainMenu.addAction("changeColour", self.drawboard.changeColour)
        # mainMenu.addAction("saveImg", self.drawboard.saveImg)
        mainMenu.addAction("start", self.startGame)
        mainMenu.addAction("clear", self.drawboard.clear_board)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout()
        central_widget.setLayout(vbox)
        vbox.addWidget(self.viewer)
        vbox.addWidget(self.drawboard)

        self.create_label()
        self.gaming = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerTimeout)

    def create_label(self):
        # label_dict = {}
        # path = r'dataset\train_simplified'
        # filenames = sorted(glob.glob(os.path.join(path, '*.csv')))
        # for i, fn in enumerate(filenames):
        #     # print(fn[:-4].split('\\'))
        #     # print(fn[:-4].split('//', '/')[-1].replace(' ', '_'))
        #     label_dict[fn[:-4].split('\\')[-1].replace(' ', '_')] = i
        # dec_dict = {v: k for k, v in label_dict.items()}


        # hard code label instead of reading bunch of files
        self.label_dict = {'The_Eiffel_Tower': 0, 'The_Great_Wall_of_China': 1, 'The_Mona_Lisa': 2, 'airplane': 3,
                       'alarm_clock': 4, 'ambulance': 5, 'angel': 6, 'animal_migration': 7, 'ant': 8, 'anvil': 9,
                       'apple': 10, 'arm': 11, 'asparagus': 12, 'axe': 13, 'backpack': 14, 'banana': 15, 'bandage': 16,
                       'barn': 17, 'baseball_bat': 18, 'baseball': 19, 'basket': 20, 'basketball': 21, 'bat': 22,
                       'bathtub': 23, 'beach': 24, 'bear': 25, 'beard': 26, 'bed': 27, 'bee': 28, 'belt': 29,
                       'bench': 30, 'bicycle': 31, 'binoculars': 32, 'bird': 33, 'birthday_cake': 34, 'blackberry': 35,
                       'blueberry': 36, 'book': 37, 'boomerang': 38, 'bottlecap': 39, 'bowtie': 40, 'bracelet': 41,
                       'brain': 42, 'bread': 43, 'bridge': 44, 'broccoli': 45, 'broom': 46, 'bucket': 47,
                       'bulldozer': 48, 'bus': 49, 'bush': 50, 'butterfly': 51, 'cactus': 52, 'cake': 53,
                       'calculator': 54, 'calendar': 55, 'camel': 56, 'camera': 57, 'camouflage': 58, 'campfire': 59,
                       'candle': 60, 'cannon': 61, 'canoe': 62, 'car': 63, 'carrot': 64, 'castle': 65, 'cat': 66,
                       'ceiling_fan': 67, 'cell_phone': 68, 'cello': 69, 'chair': 70, 'chandelier': 71, 'church': 72,
                       'circle': 73, 'clarinet': 74, 'clock': 75, 'cloud': 76, 'coffee_cup': 77, 'compass': 78,
                       'computer': 79, 'cookie': 80, 'cooler': 81, 'couch': 82, 'cow': 83, 'crab': 84, 'crayon': 85,
                       'crocodile': 86, 'crown': 87, 'cruise_ship': 88, 'cup': 89, 'diamond': 90, 'dishwasher': 91,
                       'diving_board': 92, 'dog': 93, 'dolphin': 94, 'donut': 95, 'door': 96, 'dragon': 97,
                       'dresser': 98, 'drill': 99, 'drums': 100, 'duck': 101, 'dumbbell': 102, 'ear': 103, 'elbow': 104,
                       'elephant': 105, 'envelope': 106, 'eraser': 107, 'eye': 108, 'eyeglasses': 109, 'face': 110,
                       'fan': 111, 'feather': 112, 'fence': 113, 'finger': 114, 'fire_hydrant': 115, 'fireplace': 116,
                       'firetruck': 117, 'fish': 118, 'flamingo': 119, 'flashlight': 120, 'flip_flops': 121,
                       'floor_lamp': 122, 'flower': 123, 'flying_saucer': 124, 'foot': 125, 'fork': 126, 'frog': 127,
                       'frying_pan': 128, 'garden_hose': 129, 'garden': 130, 'giraffe': 131, 'goatee': 132,
                       'golf_club': 133, 'grapes': 134, 'grass': 135, 'guitar': 136, 'hamburger': 137, 'hammer': 138,
                       'hand': 139, 'harp': 140, 'hat': 141, 'headphones': 142, 'hedgehog': 143, 'helicopter': 144,
                       'helmet': 145, 'hexagon': 146, 'hockey_puck': 147, 'hockey_stick': 148, 'horse': 149,
                       'hospital': 150, 'hot_air_balloon': 151, 'hot_dog': 152, 'hot_tub': 153, 'hourglass': 154,
                       'house_plant': 155, 'house': 156, 'hurricane': 157, 'ice_cream': 158, 'jacket': 159, 'jail': 160,
                       'kangaroo': 161, 'key': 162, 'keyboard': 163, 'knee': 164, 'ladder': 165, 'lantern': 166,
                       'laptop': 167, 'leaf': 168, 'leg': 169, 'light_bulb': 170, 'lighthouse': 171, 'lightning': 172,
                       'line': 173, 'lion': 174, 'lipstick': 175, 'lobster': 176, 'lollipop': 177, 'mailbox': 178,
                       'map': 179, 'marker': 180, 'matches': 181, 'megaphone': 182, 'mermaid': 183, 'microphone': 184,
                       'microwave': 185, 'monkey': 186, 'moon': 187, 'mosquito': 188, 'motorbike': 189, 'mountain': 190,
                       'mouse': 191, 'moustache': 192, 'mouth': 193, 'mug': 194, 'mushroom': 195, 'nail': 196,
                       'necklace': 197, 'nose': 198, 'ocean': 199, 'octagon': 200, 'octopus': 201, 'onion': 202,
                       'oven': 203, 'owl': 204, 'paint_can': 205, 'paintbrush': 206, 'palm_tree': 207, 'panda': 208,
                       'pants': 209, 'paper_clip': 210, 'parachute': 211, 'parrot': 212, 'passport': 213, 'peanut': 214,
                       'pear': 215, 'peas': 216, 'pencil': 217, 'penguin': 218, 'piano': 219, 'pickup_truck': 220,
                       'picture_frame': 221, 'pig': 222, 'pillow': 223, 'pineapple': 224, 'pizza': 225, 'pliers': 226,
                       'police_car': 227, 'pond': 228, 'pool': 229, 'popsicle': 230, 'postcard': 231, 'potato': 232,
                       'power_outlet': 233, 'purse': 234, 'rabbit': 235, 'raccoon': 236, 'radio': 237, 'rain': 238,
                       'rainbow': 239, 'rake': 240, 'remote_control': 241, 'rhinoceros': 242, 'river': 243,
                       'roller_coaster': 244, 'rollerskates': 245, 'sailboat': 246, 'sandwich': 247, 'saw': 248,
                       'saxophone': 249, 'school_bus': 250, 'scissors': 251, 'scorpion': 252, 'screwdriver': 253,
                       'sea_turtle': 254, 'see_saw': 255, 'shark': 256, 'sheep': 257, 'shoe': 258, 'shorts': 259,
                       'shovel': 260, 'sink': 261, 'skateboard': 262, 'skull': 263, 'skyscraper': 264,
                       'sleeping_bag': 265, 'smiley_face': 266, 'snail': 267, 'snake': 268, 'snorkel': 269,
                       'snowflake': 270, 'snowman': 271, 'soccer_ball': 272, 'sock': 273, 'speedboat': 274,
                       'spider': 275, 'spoon': 276, 'spreadsheet': 277, 'square': 278, 'squiggle': 279, 'squirrel': 280,
                       'stairs': 281, 'star': 282, 'steak': 283, 'stereo': 284, 'stethoscope': 285, 'stitches': 286,
                       'stop_sign': 287, 'stove': 288, 'strawberry': 289, 'streetlight': 290, 'string_bean': 291,
                       'submarine': 292, 'suitcase': 293, 'sun': 294, 'swan': 295, 'sweater': 296, 'swing_set': 297,
                       'sword': 298, 't-shirt': 299, 'table': 300, 'teapot': 301, 'teddy-bear': 302, 'telephone': 303,
                       'television': 304, 'tennis_racquet': 305, 'tent': 306, 'tiger': 307, 'toaster': 308, 'toe': 309,
                       'toilet': 310, 'tooth': 311, 'toothbrush': 312, 'toothpaste': 313, 'tornado': 314,
                       'tractor': 315, 'traffic_light': 316, 'train': 317, 'tree': 318, 'triangle': 319,
                       'trombone': 320, 'truck': 321, 'trumpet': 322, 'umbrella': 323, 'underwear': 324, 'van': 325,
                       'vase': 326, 'violin': 327, 'washing_machine': 328, 'watermelon': 329, 'waterslide': 330,
                       'whale': 331, 'wheel': 332, 'windmill': 333, 'wine_bottle': 334, 'wine_glass': 335,
                       'wristwatch': 336, 'yoga': 337, 'zebra': 338, 'zigzag': 339}
        self.dec_dict = {0: 'The_Eiffel_Tower', 1: 'The_Great_Wall_of_China', 2: 'The_Mona_Lisa', 3: 'airplane',
                     4: 'alarm_clock', 5: 'ambulance', 6: 'angel', 7: 'animal_migration', 8: 'ant', 9: 'anvil',
                     10: 'apple', 11: 'arm', 12: 'asparagus', 13: 'axe', 14: 'backpack', 15: 'banana', 16: 'bandage',
                     17: 'barn', 18: 'baseball_bat', 19: 'baseball', 20: 'basket', 21: 'basketball', 22: 'bat',
                     23: 'bathtub', 24: 'beach', 25: 'bear', 26: 'beard', 27: 'bed', 28: 'bee', 29: 'belt', 30: 'bench',
                     31: 'bicycle', 32: 'binoculars', 33: 'bird', 34: 'birthday_cake', 35: 'blackberry',
                     36: 'blueberry', 37: 'book', 38: 'boomerang', 39: 'bottlecap', 40: 'bowtie', 41: 'bracelet',
                     42: 'brain', 43: 'bread', 44: 'bridge', 45: 'broccoli', 46: 'broom', 47: 'bucket', 48: 'bulldozer',
                     49: 'bus', 50: 'bush', 51: 'butterfly', 52: 'cactus', 53: 'cake', 54: 'calculator', 55: 'calendar',
                     56: 'camel', 57: 'camera', 58: 'camouflage', 59: 'campfire', 60: 'candle', 61: 'cannon',
                     62: 'canoe', 63: 'car', 64: 'carrot', 65: 'castle', 66: 'cat', 67: 'ceiling_fan', 68: 'cell_phone',
                     69: 'cello', 70: 'chair', 71: 'chandelier', 72: 'church', 73: 'circle', 74: 'clarinet',
                     75: 'clock', 76: 'cloud', 77: 'coffee_cup', 78: 'compass', 79: 'computer', 80: 'cookie',
                     81: 'cooler', 82: 'couch', 83: 'cow', 84: 'crab', 85: 'crayon', 86: 'crocodile', 87: 'crown',
                     88: 'cruise_ship', 89: 'cup', 90: 'diamond', 91: 'dishwasher', 92: 'diving_board', 93: 'dog',
                     94: 'dolphin', 95: 'donut', 96: 'door', 97: 'dragon', 98: 'dresser', 99: 'drill', 100: 'drums',
                     101: 'duck', 102: 'dumbbell', 103: 'ear', 104: 'elbow', 105: 'elephant', 106: 'envelope',
                     107: 'eraser', 108: 'eye', 109: 'eyeglasses', 110: 'face', 111: 'fan', 112: 'feather',
                     113: 'fence', 114: 'finger', 115: 'fire_hydrant', 116: 'fireplace', 117: 'firetruck', 118: 'fish',
                     119: 'flamingo', 120: 'flashlight', 121: 'flip_flops', 122: 'floor_lamp', 123: 'flower',
                     124: 'flying_saucer', 125: 'foot', 126: 'fork', 127: 'frog', 128: 'frying_pan', 129: 'garden_hose',
                     130: 'garden', 131: 'giraffe', 132: 'goatee', 133: 'golf_club', 134: 'grapes', 135: 'grass',
                     136: 'guitar', 137: 'hamburger', 138: 'hammer', 139: 'hand', 140: 'harp', 141: 'hat',
                     142: 'headphones', 143: 'hedgehog', 144: 'helicopter', 145: 'helmet', 146: 'hexagon',
                     147: 'hockey_puck', 148: 'hockey_stick', 149: 'horse', 150: 'hospital', 151: 'hot_air_balloon',
                     152: 'hot_dog', 153: 'hot_tub', 154: 'hourglass', 155: 'house_plant', 156: 'house',
                     157: 'hurricane', 158: 'ice_cream', 159: 'jacket', 160: 'jail', 161: 'kangaroo', 162: 'key',
                     163: 'keyboard', 164: 'knee', 165: 'ladder', 166: 'lantern', 167: 'laptop', 168: 'leaf',
                     169: 'leg', 170: 'light_bulb', 171: 'lighthouse', 172: 'lightning', 173: 'line', 174: 'lion',
                     175: 'lipstick', 176: 'lobster', 177: 'lollipop', 178: 'mailbox', 179: 'map', 180: 'marker',
                     181: 'matches', 182: 'megaphone', 183: 'mermaid', 184: 'microphone', 185: 'microwave',
                     186: 'monkey', 187: 'moon', 188: 'mosquito', 189: 'motorbike', 190: 'mountain', 191: 'mouse',
                     192: 'moustache', 193: 'mouth', 194: 'mug', 195: 'mushroom', 196: 'nail', 197: 'necklace',
                     198: 'nose', 199: 'ocean', 200: 'octagon', 201: 'octopus', 202: 'onion', 203: 'oven', 204: 'owl',
                     205: 'paint_can', 206: 'paintbrush', 207: 'palm_tree', 208: 'panda', 209: 'pants',
                     210: 'paper_clip', 211: 'parachute', 212: 'parrot', 213: 'passport', 214: 'peanut', 215: 'pear',
                     216: 'peas', 217: 'pencil', 218: 'penguin', 219: 'piano', 220: 'pickup_truck',
                     221: 'picture_frame', 222: 'pig', 223: 'pillow', 224: 'pineapple', 225: 'pizza', 226: 'pliers',
                     227: 'police_car', 228: 'pond', 229: 'pool', 230: 'popsicle', 231: 'postcard', 232: 'potato',
                     233: 'power_outlet', 234: 'purse', 235: 'rabbit', 236: 'raccoon', 237: 'radio', 238: 'rain',
                     239: 'rainbow', 240: 'rake', 241: 'remote_control', 242: 'rhinoceros', 243: 'river',
                     244: 'roller_coaster', 245: 'rollerskates', 246: 'sailboat', 247: 'sandwich', 248: 'saw',
                     249: 'saxophone', 250: 'school_bus', 251: 'scissors', 252: 'scorpion', 253: 'screwdriver',
                     254: 'sea_turtle', 255: 'see_saw', 256: 'shark', 257: 'sheep', 258: 'shoe', 259: 'shorts',
                     260: 'shovel', 261: 'sink', 262: 'skateboard', 263: 'skull', 264: 'skyscraper',
                     265: 'sleeping_bag', 266: 'smiley_face', 267: 'snail', 268: 'snake', 269: 'snorkel',
                     270: 'snowflake', 271: 'snowman', 272: 'soccer_ball', 273: 'sock', 274: 'speedboat', 275: 'spider',
                     276: 'spoon', 277: 'spreadsheet', 278: 'square', 279: 'squiggle', 280: 'squirrel', 281: 'stairs',
                     282: 'star', 283: 'steak', 284: 'stereo', 285: 'stethoscope', 286: 'stitches', 287: 'stop_sign',
                     288: 'stove', 289: 'strawberry', 290: 'streetlight', 291: 'string_bean', 292: 'submarine',
                     293: 'suitcase', 294: 'sun', 295: 'swan', 296: 'sweater', 297: 'swing_set', 298: 'sword',
                     299: 't-shirt', 300: 'table', 301: 'teapot', 302: 'teddy-bear', 303: 'telephone',
                     304: 'television', 305: 'tennis_racquet', 306: 'tent', 307: 'tiger', 308: 'toaster', 309: 'toe',
                     310: 'toilet', 311: 'tooth', 312: 'toothbrush', 313: 'toothpaste', 314: 'tornado', 315: 'tractor',
                     316: 'traffic_light', 317: 'train', 318: 'tree', 319: 'triangle', 320: 'trombone', 321: 'truck',
                     322: 'trumpet', 323: 'umbrella', 324: 'underwear', 325: 'van', 326: 'vase', 327: 'violin',
                     328: 'washing_machine', 329: 'watermelon', 330: 'waterslide', 331: 'whale', 332: 'wheel',
                     333: 'windmill', 334: 'wine_bottle', 335: 'wine_glass', 336: 'wristwatch', 337: 'yoga',
                     338: 'zebra', 339: 'zigzag'}

    def startGame(self):
        self.gaming = True
        self.drawboard.clear_board()
        label = self.dec_dict[random.randrange(len(self.label_dict) + 1)]
        self.update_label(label)
        self.timeLeft = 20
        self.timer.start(1000)
        self.update_timer()
        self.label = label


    def timerTimeout(self):
        if self.gaming:
            # if self.timeLeft % 4 == 0 or self.timeLeft < 3:

            p = ProcessRunnable(target=self.drawboard.compute, args=())
            p.start()
            print('timeout here: {}'.format(self.timeLeft))

            # label3 = self.drawboard.compute()
            # if self.label in label3:
            #     self.endGame(0)


            self.timeLeft -= 1
            if self.timeLeft == 0:
                self.endGame(0)

            self.update_timer()

    def endGame(self, win):
        print('label is {}'.format(self.label))
        if win:
            self.update_label('WIN, TIME LEFT: ')
            # self.update_label('WIN, TIME LEFT {}'.format(str(self.timeLeft)))
        else:
            self.update_label('LOSE')
        self.gaming = False

    def update_timer(self):
        self.time_counter.setText(str(self.timeLeft))

    def update_label(self, label):
        self.guess_label.setText(label)

    def update_top3(self, labels):
        self.top3_label.setText('Are you drawing {} ? {} ? {} ?'.format(labels[0], labels[1], labels[2]))
        # print(self.menuBar().size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()