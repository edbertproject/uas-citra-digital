from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from butterworth import Butter
import sys
import cv2
import imutils
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('qt-image-processing.ui', self)
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))

        self.originalImage = None
        self.image = None
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionPrint.triggered.connect(self.print_dialog)
        self.actionQuit.triggered.connect(self.quit_dialog)
        self.actionAuthor.triggered.connect(self.author_dialog)

        # right bar
        self.rotateDial.valueChanged.connect(self.rotate)
        self.scaleSlider.valueChanged.connect(self.scale)
        self.brightnessSlider.valueChanged.connect(self.brightness)
        self.contrastSlider.valueChanged.connect(self.scale)

        # left bar
        self.rgbButton.clicked.connect(self.rgb)
        self.hsvButton.clicked.connect(self.hsv)
        self.labButton.clicked.connect(self.lab)
        self.grayscaleButton.clicked.connect(self.grayscale)
        self.resetButton.clicked.connect(self.reset)

    @pyqtSlot()
    def load_image(self, fname):
        self.originalImage = imutils.resize(cv2.imread(fname),width=500) # for efficient processing
        self.image = self.originalImage
        self.display_image()

    def display_image(self):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] is the number of pixels in the Y direction
        # image.shape[1] is the number of pixels in the x direction
        # image.shape[2] store the number of channels representing each pixel
        img = img.rgbSwapped()  # efficiently convert an RGB image to a BGR image.
        self.imageCanvas.setPixmap(QPixmap.fromImage(img))
        self.imageCanvas.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Align the position of the image on the label

    def open_image(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\\Edbert\\Pictures',"Image Files (*)")
        if fname:
            self.load_image(fname)
        else:
            print("Invalid Image")

    def save_image(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\Edbert\\Pictures', "Image Files (*.bmp)")
        if fname:
            cv2.imwrite(fname, self.image)
            print("Error")

    def print_dialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imageCanvas.print_(printer)

    def reset(self):
        self.image = self.originalImage
        self.display_image()

    def author_dialog(self):
        QMessageBox.about(self, "About Author", "Image Processing Apps\n\n"
                                                "32190037 – Effendy\n"
                                                "32190047 – Edbert Andoyo\n"
                                                "32190086 – Adam Khadafi\n"
                          )

    def quit_dialog(self):
        message = QMessageBox.question(self, "Exit", "Are you sure want to exit?", QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

    def rgb(self):
        self.image = self.originalImage
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image()

    def hsv(self):
        self.image = self.originalImage
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.display_image()

    def lab(self):
        self.image = self.originalImage
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.display_image()

    def grayscale(self):
        self.image = self.originalImage
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.display_image()

    def rotate(self, angle):
        self.image = self.originalImage
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.display_image()

    def scale(self, c):
        self.image = self.originalImage
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.display_image()

    def brightness(self, c):
        self.image = self.originalImage
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - c
        v[v > lim] = 255
        v[v <= lim] += c

        final_hsv = cv2.merge((h, s, v))
        self.image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.display_image()


app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())