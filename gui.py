import sys
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QMimeData, QPointF, Qt, QObject, pyqtSlot, QSize, QAbstractListModel, QRectF, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QDrag, QPainter, QStandardItemModel, QIcon, QPen, QColor, QCursor
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QGridLayout,
                             QLabel, QPushButton, QWidget, QVBoxLayout, QListWidget, QAbstractItemView, QHBoxLayout,
                             QListView, QListWidgetItem, QMainWindow, QStackedWidget, QStackedLayout, QMenu, QMenuBar,
                             QAction, QSpacerItem, QSizePolicy, QSlider)

import pandas as pd
from simplification.cutil import simplify_coords

class DrawBoard(QLabel):
    def __init__(self):
        super(DrawBoard, self).__init__()
        self.setFixedSize(QSize(800, 600))
        print(self.size())
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        self.image.fill(Qt.white)
        self.imageDraw = QImage(self.size(), QImage.Format_ARGB32)
        # self.imageDraw.fill(Qt.transparent)
        self.imageDraw.fill(Qt.white)

        self.drawing = False
        self.brushSize = 2
        self._clear_size = 20
        self.brushColor = QColor(Qt.black)
        self.lastPoint = QPoint()
        self.change = False
        self.strokes = []
        self.stroke = []

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
        # canvasPainter = QPainter(self)
        canvasPainter = QPainter(self)
        # canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        canvasPainter.drawImage(self.rect(), self.imageDraw, self.imageDraw.rect())

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

    def saveImg(self):
        # print(123)
        qimg = self.imageDraw
        # print(456)
        qimg.save('images/tmp.png', 'png')
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
        mx = 0
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
            print(a, b)
            p = np.stack((a, b), axis=-1)
            print(p)
            print(p.shape)
            # breakpoint()

            p = simplify_coords(p, 2.0)
            p = np.split(p, [-1], axis=1)
            print(p)

            p = np.array(p).squeeze(-1).astype(np.uint8)
            stks.append(p.tolist())


        data = pd.DataFrame([[9000003627287624, 'UA', stks.__str__()]], columns=['key_id', 'countrycode', 'drawing'])
        data.to_csv('dataset/test_simplified/tmp.csv')

        # img = self.draw(stks)
        # cv.imwrite('images/sim_tmp.png', img)
        self.compute()

    def compute(self):
        # compute label








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

        print(self.drawboard.size())
        self.time_counter = QLabel()

        mainMenu = self.menuBar()
        # mainMenu.addAction("changeColour", self.drawboard.changeColour)
        mainMenu.addAction("saveImg", self.drawboard.saveImg)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout()
        central_widget.setLayout(vbox)
        vbox.addWidget(self.drawboard)


        # print(self.menuBar().size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()