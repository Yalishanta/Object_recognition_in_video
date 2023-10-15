import cv2

# Создаем объект класса CascadeClassifier для распознавания лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Создаем объект класса CascadeClassifier для распознавания тел животных
animal_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Задаем номер камеры (если их несколько)
cap = cv2.VideoCapture(0)

# Инициализируем переменную для подсчета количества людей
people_count = 0

while True:
    # Получаем изображение с камеры
    ret, frame = cap.read()

    # Если изображение получено успешно
    if ret:
        # Преобразуем изображение в черно-белое
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Распознаем лица на изображении и создаем прямоугольник вокруг каждого лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Распознаем тела животных на изображении и создаем прямоугольник вокруг каждого тела
        animals = animal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in animals:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Подсчитываем количество людей на изображении и добавляем их к общему количеству
        people_count += len(faces)

        # Выводим количество людей на изображении
        cv2.putText(frame, f"People: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображаем изображение
        cv2.imshow('frame', frame)

    # Если изображение не получено успешно - выходим из цикла
    else:
        break

    # Если нажата клавиша 'q' - выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
