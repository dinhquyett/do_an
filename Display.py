import cv2
import tensorflow as tf
import numpy as np


# Khởi tạo bộ nhận diện khuôn mặt và mô hình đã huấn luyện
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
save_model = tf.keras.models.load_model("khuonmat1.keras")

# Mở video capture device (camera)
cap = cv2.VideoCapture(0)  # 0: camera mặc định
# filename = 'test/q2.jpg'
# image = cv2.imread(filename)
while True:
    # Đọc frame từ video capture device
    ret, frame = cap.read()

    # Chuyển đổi frame sang ảnh xám để nhận diện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Vòng lặp qua từng khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hộp bao quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Cắt và chuẩn bị ảnh khuôn mặt để đưa vào mô hình dự đoán
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
        roi_gray = roi_gray.reshape((100, 100, 1))
        roi_gray = np.array(roi_gray)

        # Dự đoán kết quả
        result = save_model.predict(np.array([roi_gray]))
        final = np.argmax(result)

        # Hiển thị tên của người được dự đoán lên ảnh
        if final == 0:
            cv2.putText(frame, "Tran Thanh", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 1:
            cv2.putText(frame, "Truong Giang", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 2:
            cv2.putText(frame, "Hoai Linh", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 3:
            cv2.putText(frame, "Son Tung", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif final == 4:
            cv2.putText(frame, "Dinh Quyet", (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị video với khuôn mặt đã nhận diện
    cv2.imshow('Face Recognition', frame)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng video capture và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()