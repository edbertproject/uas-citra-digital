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

# =====================================================================================
# 
class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('qt-image-processing.ui', self)

        # Apps Icon
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))

        # Declare variable
        self.originalImage = None
        self.image = None
        
        # init Toolbar and Menubar button
        self.actionOpen.triggered.connect(self.open_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionPrint.triggered.connect(self.print_dialog)
        self.actionQuit.triggered.connect(self.quit_dialog)
        self.actionAuthor.triggered.connect(self.author_dialog)

# ===========================================================================================
        # Right Bar
        self.rotateDial.valueChanged.connect(self.rotate)
        self.scaleSlider.valueChanged.connect(self.scale)
        self.brightnessSlider.valueChanged.connect(self.brightness)
        self.contrastSlider.valueChanged.connect(self.scale)
        self.countButton.clicked.connect(self.count)
        # self.chainCodeButton.valueChanged.connect(self.chainCode)

# ==========================================================================================
        # Left Bar
        # Color Space
        self.rgbButton.clicked.connect(self.rgb)
        self.hsvButton.clicked.connect(self.hsv)
        self.labButton.clicked.connect(self.lab)
        self.grayscaleButton.clicked.connect(self.grayscale)
        self.inversButton.clicked.connect(self.invers)
        
        # Filtering
        self.smoothButton.clicked.connect(self.smooth)
        self.sharpButton.clicked.connect(self.sharp)

        # Edge Detection
        self.prewittButton.clicked.connect(self.prewitt)
        self.sobelButton.clicked.connect(self.sobel)
        self.robertsButton.clicked.connect(self.roberts)
        self.boundingBoxButton.clicked.connect(self.boundingBox)
        
        # Center of Gravity
        # self.centerGravityButton.clicked.connect(self.smooth)
        

        # Reset Image
        self.resetButton.clicked.connect(self.reset)

        if self.originalImage == None:
            self.actionSave.setEnabled(False)
            self.actionPrint.setEnabled(False)

            self.rgbButton.setEnabled(False)
            self.hsvButton.setEnabled(False)
            self.labButton.setEnabled(False)
            self.grayscaleButton.setEnabled(False)
            self.inversButton.setEnabled(False)

            self.smoothButton.setEnabled(False)
            self.sharpButton.setEnabled(False)

            self.prewittButton.setEnabled(False)
            self.sobelButton.setEnabled(False)
            self.robertsButton.setEnabled(False)
            self.boundingBoxButton.setEnabled(False)

            self.rotateDial.setEnabled(False)
            self.scaleSlider.setEnabled(False)
            self.brightnessSlider.setEnabled(False)
            self.contrastSlider.setEnabled(False)
    

            self.centerGravityButton.setEnabled(False)

            self.resetButton.setEnabled(False)

            self.countButton.setEnabled(False)

            self.chainCodeButton.setEnabled(False)
        

# =======================================================================================================
# Displaying Image Method
    @pyqtSlot()
    def load_image(self, fname):
        self.originalImage = imutils.resize(cv2.imread(fname),width=500) # for efficient processing
        self.image = self.originalImage
        self.display_image()

        self.actionSave.setEnabled(True)
        self.actionPrint.setEnabled(True)

        self.rgbButton.setEnabled(True)
        self.hsvButton.setEnabled(True)
        self.labButton.setEnabled(True)
        self.grayscaleButton.setEnabled(True)
        self.inversButton.setEnabled(True)
        
        self.smoothButton.setEnabled(True)
        self.sharpButton.setEnabled(True)

        self.prewittButton.setEnabled(True)
        self.sobelButton.setEnabled(True)
        self.robertsButton.setEnabled(True)
        self.boundingBoxButton.setEnabled(True)

        self.rotateDial.setEnabled(True)
        self.scaleSlider.setEnabled(True)
        self.brightnessSlider.setEnabled(True)
        self.contrastSlider.setEnabled(True)
    
        self.centerGravityButton.setEnabled(True)

        self.resetButton.setEnabled(True)

        self.countButton.setEnabled(True)

        self.chainCodeButton.setEnabled(True)
        

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

    # Filling Method
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

    # Reset Method
    def reset(self):
        self.image = self.originalImage
        self.display_image()

    # About Us Method
    def author_dialog(self):
        QMessageBox.about(self, "About Author", "Image Processing Apps\n\n"
                                                "32190037 – Effendy\n"
                                                "32190047 – Edbert Andoyo\n"
                                                "32190086 – Adam Khadafi\n"
                          )
    # Quit Method
    def quit_dialog(self):
        message = QMessageBox.question(self, "Exit", "Are you sure want to exit?", QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

# ==========================================================================================================
    # Color Space Method
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

    def invers(self):
        self.image = self.originalImage
        self.image = 255 - self.image
        self.display_image()

# Rotate and Scale
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


# Brightness and Contrast
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

# Filtering
    def smooth(self):
        self.image = self.originalImage
        self.image = cv2.blur(self.image,(5,5))
        self.display_image()
    
    def sharp(self):
        self.image = self.originalImage
        kernel =  np.array([[0, -1, 0], 
                            [-1, 5, -1], 
                            [0, -1, 0]])
        self.image = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        self.display_image()

# Edge Detection
    def boundingBox(self):
        self.image = self.originalImage
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        x,y,w,h = cv2.boundingRect(thresh)
        
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 2)
        self.display_image()

    def prewitt(self):
        self.image = self.originalImage
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        kernelx = np.array([[1, 0, -1], 
                            [1, 0, -1],
                            [1, 0, -1]], dtype=int)
        kernely = np.array([[1, 1, 1], 
                            [0, 0, 0],
                            [-1, -1, -1]], dtype=int)

        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        self.image = cv2.addWeighted(absX, 1, absY, 1, 0)

        self.display_image()

    def sobel(self):
        self.image = self.originalImage
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        kernelx = np.array([[1, 0, -1], 
                            [2, 0, -2],
                            [1, 0, -1]], dtype=int)
        kernely = np.array([[1, 2, 1], 
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=int)

        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        self.image = cv2.addWeighted(absX, 1, absY, 1, 0)

        self.display_image()

    def roberts(self):
        self.image = self.originalImage
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        kernelx = np.array([[-1, 0], 
                            [0, 1]], dtype=int)
        kernely = np.array([[0, -1], 
                            [1, 0]], dtype=int)

        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        self.image = cv2.addWeighted(absX, 1, absY, 1, 0)

        self.display_image()

# Area and Perimeter
    def count(self):
        self.image = self.originalImage
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        x,y,w,h = cv2.boundingRect(thresh)
        
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 2)

        panjang = w - x
        lebar = h - y

        # Connverting pixel to cm
        panjang = panjang * 0.03
        lebar = lebar * 0.03

        #  Declare Luas and Keliling Variable
        luas = panjang*lebar
        keliling = 2*(panjang+lebar)

        self.luasEdit.setText(str(round(luas)))
        self.kelilingEdit.setText(str(round(keliling)))

        self.display_image()

# ===============================================================================================
app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())
