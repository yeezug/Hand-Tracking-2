import cv2
import mediapipe as mp
import time
import math
import keyboard

class handDetector(): # Этот класс отвечает за обнаружение рук на изображении и извлечение информации о их положении и форме.
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5): # Это конструктор класса.
        self.mode = mode # Устанавливает режим работы модели MediaPipe для обнаружения рук.  `False` означает  "статический" режим, то есть модель ищет руки на каждом кадре независимо от предыдущих кадров.
        self.maxHands = maxHands #Определяет максимальное количество рук, которое нужно обнаружить (по умолчанию 2)
        self.modelComplex = modelComplexity # Устанавливает уровень сложности модели.  `1` - средняя сложность.
        self.detectionCon = detectionCon # Устанавливает порог уверенности для обнаружения рук.  Значение `0.5` означает, что модель должна быть уверена на 50% в том, что она нашла руку.
        self.trackCon = trackCon # Устанавливает порог уверенности для отслеживания рук.  Это важно для  "динамического" режима, когда модель  следит за движением рук.

        self.mpHands = mp.solutions.hands #Загружает модель MediaPipe для обнаружения рук.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon) # Создает объект модели MediaPipe с указанными настройками.
        self.mpDraw = mp.solutions.drawing_utils # Загружает инструменты MediaPipe для рисования результатов.
        
        self.tipIds = [4, 8, 12, 16, 20] #Список индексов точек, которые  представляют кончики пальцев.


    def findHands(self, img, draw = True): # Эта функция  обрабатывает изображение, чтобы найти руки.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Преобразует изображение из BGR (формат OpenCV) в RGB (формат, который  использует MediaPipe).
        self.results = self.hands.process(imgRGB) # Пропускает изображение через модель MediaPipe, чтобы найти руки.  Результаты сохраняются в  `self.results`.
        
        
        if self.results.multi_hand_landmarks: # Если модель обнаружила руки, то:
            for handLms in self.results.multi_hand_landmarks: # Перебирает все найденные руки.
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS) #Рисует  скелет руки на изображении, используя информацию о ключевых точках.
        return img
            
    def findPosition(self, img, handNo = 0, draw = True): #Эта функция получает  координаты  ключевых  точек руки.
        
        self.lmList = [] #Создает пустой список для хранения координат.
        if self.results.multi_hand_landmarks: #  Если модель обнаружила руки, то:
            myHand = self.results.multi_hand_landmarks[handNo] # Выбирает нужную руку, если их несколько.
            
            for id, lm in enumerate(myHand.landmark): # Перебирает все  ключевые  точки руки.
                h, w, c = img.shape # Получает высоту, ширину и количество каналов изображения.
                cx, cy = int(lm.x * w), int(lm.y * h) # Преобразует относительные координаты ключевой  точки  (от 0 до 1) в абсолютные координаты  в пикселях.
                self.lmList.append([id, cx, cy]) # Сохраняет номер  ключевой  точки (id) и ее координаты (cx, cy) в список `self.lmList`.
                if draw: #Если  `draw`  установлен в `True`,  то  рисует кружок  вокруг  каждой ключевой  точки.
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return self.lmList
    
    def fingersUp(self, ): #Эта функция  определяет, какие  пальцы подняты.
        fingers = [] #Создает  пустой список для  хранения  информации о  поднятых пальцах.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]: # Проверяет, поднят ли большой палец.
            fingers.append(1) 
        else:
            fingers.append(0)
        for id in range(1, 5): #Перебирает остальные пальцы.
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]: # Проверяет, поднят ли палец (кончик пальца должен быть выше, чем предыдущая точка).
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3): #Эта функция  вычисляет  расстояние между двумя  ключевыми  точками.
        x1, y1 = self.lmList[p1][1:] # Получает координаты первой точки.
        x2, y2 = self.lmList[p2][1:] # Получает координаты второй точки.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Вычисляет  координаты  центра  линии  между точками.

        if draw: # Если  `draw`  установлен в `True`,  то:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t) # Рисует  линию  между точками.
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED) #Рисует  кружки  вокруг  точек.
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED) # Рисует  круг  в  центре  линии.
            length = math.hypot(x2 - x1, y2 - y1) #Вычисляет расстояние между  точками  (гипотенузу прямоугольного треугольника).

        return length, img, [x1, y1, x2, y2, cx, cy]
    
def main(): #Эта функция  выполняет главную  логику  программы.
    pTime = 0 
    cTime = 0 #Переменные для хранения  времени  в  начале  и  в конце  кадра  для  вычисления FPS.
    cap = cv2.VideoCapture("2.mp4") #Создает  объект  `VideoCapture`, который  загружает  видеофайл  "2.mp4".
    detector = handDetector() #Создает  объект  класса  `handDetector`.
    while True: #Создает  бесконечный цикл  для  обработки  кадров  видео.
        success, img = cap.read() #Считывает  следующий  кадр  из  видеофайла.  `success`  -  флаг,  указывающий,  был ли  кадр  считан  успешно.
        img = detector.findHands(img) #Обнаруживает  руки  на  кадре  и  рисует  их.
        lmList = detector.findPosition(img) #Получает  координаты  ключевых  точек  рук.
        if len(lmList) != 0: #Если  обнаружены  руки,  то  выводит  координаты  точки  4 (кончика  большого  пальца).
            print(lmList[4])
        
        cTime = time.time() #Получает  текущее  время.
        fps = 1/(cTime-pTime) #Вычисляет  FPS  (количество  кадров  в  секунду).
        pTime = cTime # Обновляет  время  предыдущего  кадра.
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, 
                    (255, 0, 255), 3) #Отображает  FPS  на  изображении.
        
        cv2.imshow("Image", img) # Отображает  обработанный  кадр  в  окне.
        cv2.waitKey(1) #Ожидает  нажатия  клавиши  в  течение  1  миллисекунды.  Это  необходимо  для  обновления  изображения.
        if keyboard.is_pressed("space"):  #Проверяет,  была  ли  нажата  клавиша  "Пробел".  Если  да,  то  прерывает  цикл  обработки  видео.
            break
    
if __name__ == "__main__":#Эта  строка  кода  обеспечивает  выполнение  функции  `main`  только  в  том  случае,  если  файл  запускается  непосредственно,  а  не  импортируется  как  модуль.
    main() #Вызывает  функцию  `main`.