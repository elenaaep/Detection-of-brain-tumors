import cv2
from matplotlib import pylab
from PIL import Image, ImageDraw
from numpy import average
from tensorflow import keras
import cv2 as cv
import imutils
import os

# setam mediul de lucru
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = keras.models.load_model('brain_tumor_detector.h5')  # este un fisier model

# Gasim punctele extreme
extLeft = tuple()
extRight = tuple()
extTop = tuple()
extBot = tuple()


# creez o functie care ne afiseaza rezultat
def createAndDisplayResultImage(imageName, textToInsertOnImage, res):
    img = Image.open(imageName, 'r')
    img_w, img_h = img.size
    background = Image.new('RGBA', (1000, 800),
                           (255, 255, 255, 255))  # creez un nou fundal cu anumite dimensiuni si o anumita scara
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.save('result.png')  # salvam fundalul intr-o noua imagine, pentru a-l putea vizualiza

    # Adaugam diagnosticul pe imagine(textul specific)
    image = cv.imread('result.png')  # citim imaginea salvata mai sus
    cv.putText(image, text=textToInsertOnImage, org=(10, 80),
               # la imaginea fundal, adaugam un text pe un anumit font, culoare.dimensiune etc.
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(2, 5, 5),
               thickness=2, lineType=cv.LINE_AA)

    pylab.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    pylab.show()  # vizualizam aceasta imagine

    return


# definim o functie care gaseste tumoarea
def findTumor(imageName):
    image = cv.imread(imageName, 1)  # citim imaginea
    image_contur = cv.imread(imageName, 1)  # citim imaginea care ne afiseaza conturul
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # o convertim la Grayscale
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    av = average(gray)

    # Folosim binarizarea Threshold asupra imaginii, apoi efectuam o serie de eroziuni si
    # dilatari pentru a elimina orice regiune mica de zgomot
    thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    thresh_display = cv.threshold(gray, av+80, 255, cv.THRESH_BINARY)[1]  #folosim metoda de binarizare threshold

    # Gasim conturul tumorii in imaginea binarizata
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts_display = cv.findContours(thresh_display.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #din imafinea afisata ne gaseste conturul
    cnts_display = imutils.grab_contours(cnts_display)
    c = max(cnts, key=cv.contourArea)

    image_contur = image

    # Gasim punctele extreme
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Decupam noua imagine din imaginea originala si folosim cele 4 puncte de extrem(stranga, dreapta, sus, jos)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    # imaginea o redimensionam cu imaginea noua si parametri noi
    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 240, 240, 3))  # remodeleaza imaginea

    res = model.predict(image)

    if res > 0.5:  # daca tumoarea este mai mare ca 0.5 at rezultatul este pozitiv
        image_contur = cv2.drawContours(image_contur, cnts_display, -1, (0, 255, 75), 2) #in imaginea contur ne va colora conturul tumorii
        cv2.imshow('contur', image_contur) #vom afisa aceasta imagine
        print(f'Rezultat pozitiv - Tumoare cerebrala detectata!') #rez va fi afisat in consola
        createAndDisplayResultImage(imageName, "Tumoare detectata",res)
    else:  # daca tumoarea este mai mica rezultatul este negativ
        print(f'Rezultat negativ - Nicio tumoare cerebrala detectata!')
        createAndDisplayResultImage(imageName, "Nicio tumoare detectata",res)
    return  # se returneaza in consola daca este pozitiv sau nu


# definim o functie de test
def _test(image_file):
    findTumor(image_file)  # testeaza functia care gaseste tumoarea
    return


if __name__ == '__main__':
    _test("dataset/with_tumor/brainTumor.png")
    _test("dataset/with_tumor/withTumor1.jpg")
    _test("dataset/with_tumor/Brain7.jpg")
    _test("dataset/with_tumor/brain.jpg")
    _test("dataset/with_tumor/Brain2.jpg")
    _test("dataset/no_tumor/1 no.jpeg")
    _test("dataset/no_tumor/2 no.jpeg")
    _test("dataset/no_tumor/3 no.jpg")
