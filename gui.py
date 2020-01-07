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
        label_dict = {}
        path = r'dataset\train_simplified'
        filenames = sorted(glob.glob(os.path.join(path, '*.csv')))
        for i, fn in enumerate(filenames):
            # print(fn[:-4].split('\\'))
            # print(fn[:-4].split('//', '/')[-1].replace(' ', '_'))
            label_dict[fn[:-4].split('\\')[-1].replace(' ', '_')] = i
        dec_dict = {v: k for k, v in label_dict.items()}

        self.label_dict = label_dict
        self.dec_dict = dec_dict



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

        # painter = QPainter(self.imageDraw)
        # # painter = QPainter(self)
        # painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # r = QRect(self.imageDraw.size())
        # # r.moveCenter(event.pos())
        # painter.save()
        # painter.setCompositionMode(QPainter.CompositionMode_Clear)
        # painter.eraseRect(r)
        # painter.restore()

    # def changeColour(self):
    #     self.change = not self.change
    #     if self.change:
    #         pixmap = QPixmap(QSize(1, 1) * self._clear_size)
    #         pixmap.fill(Qt.transparent)
    #         painter = QPainter(pixmap)
    #         painter.setPen(QPen(Qt.black, 2))
    #         painter.drawRect(pixmap.rect())
    #         painter.end()
    #         cursor = QCursor(pixmap)
    #         QApplication.setOverrideCursor(cursor)
    #     else:
    #         QApplication.restoreOverrideCursor()

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
        # qimg = self.imageDraw
        # print(456)
        # qimg.save('images/tmp.png', 'png')
        # mat = cv.imread('images/tmp.png', 0)
        # mat = cv.resize(mat, (224, 224))
        # print(mat)
        # print(mat.shape)
        # unique, counts = np.unique(mat, return_counts=True)
        # print(dict(zip(unique, counts)))

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

            # breakpoint()
        # print(x)
        # print(np.array(x).flatten())
        # breakpoint()
        # print(x_min, x_max, y_min, y_max)
        # breakpoint()
        stks = []
        for stk in self.strokes:
            # print(stk)
            # print(p.shape)
            p = np.split(np.array(stk), [-1], axis=1)
            p = np.array(p).astype(np.float32).squeeze(-1)

            p[0] -= x_min
            p[1] -= y_min
            p[0] *= 255.0 / (x_max - x_min)
            p[1] *= 255.0 / (y_max - y_min)
            # print(p)
            # print(p.shape)
            p = p.astype(np.uint8)
            # print(p)
            # breakpoint()
            # print(p.shape)
            a = np.array(p[0])
            b = np.array(p[1])
            # print(a, b)
            p = np.stack((a, b), axis=-1)
            # print(p)
            # print(p.shape)
            # breakpoint()

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
        label_dict = {}
        path = r'dataset\train_simplified'
        filenames = sorted(glob.glob(os.path.join(path, '*.csv')))
        for i, fn in enumerate(filenames):
            # print(fn[:-4].split('\\'))
            # print(fn[:-4].split('//', '/')[-1].replace(' ', '_'))
            label_dict[fn[:-4].split('\\')[-1].replace(' ', '_')] = i
        dec_dict = {v: k for k, v in label_dict.items()}

        self.label_dict = label_dict
        self.dec_dict = dec_dict

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