import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Khởi tạo danh sách dữ liệu và nhãn
data = []
label = []

# Đường dẫn tới thư mục chứa dữ liệu
data_directory = './dataset2/'

# Duyệt qua từng tập dữ liệu và nhãn
for j in range(1, 4):  # Số lượng lớp, chỉ có 2 trong trường hợp này
    for i in range(1, 201):  # Số lượng ảnh cho mỗi lớp, thay đổi tùy theo tập dữ liệu
        filename = os.path.join(data_directory, f'anh.{j}.{i}.jpg')

        # Kiểm tra xem tệp tồn tại không trước khi đọc
        if os.path.isfile(filename):
            Img = cv2.imread(filename)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            Img = cv2.resize(src=Img, dsize=(100, 100))
            Img = np.array(Img)
            data.append(Img)
            label.append(j - 1)
        else:
            print(f"File {filename} không tồn tại")

# Chuyển danh sách thành mảng numpy
data = np.array(data)
label = np.array(label)

# Kiểm tra xem label có dữ liệu không trước khi tiến hành chuyển đổi
if len(label) == 0:
    print("Không có dữ liệu để chuyển đổi")
else:
    # Reshape data
    data = data.reshape((-1, 100, 100, 1))

    # Normalize data
    data = data / 255.0

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(label)

# Split data into training and testing sets (if needed)

# Data augmentation
augmentor = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                               zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")

# Tạo mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(100, 100, 1)),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(5, activation="softmax")
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

# Print model summary
model.summary()

# Huấn luyện mô hình
history = model.fit(
    augmentor.flow(data, labels, batch_size=32),
    steps_per_epoch=len(data) // 32,  # Số lượng batch mỗi epoch
    epochs=20,  # Số lượng epochs
    verbose=1
)

# Lưu mô hình
model.save("khuonmat3.keras")
