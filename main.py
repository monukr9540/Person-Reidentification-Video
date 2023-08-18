#importing libraries
import numpy as np
import pandas as pd
import cv2
import os
import csv
import PIL
import shutil
import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img

#detecting people and saving them in a directory->

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

video_path = 'sample_video_1.mp4'
cap = cv2.VideoCapture(video_path)

output_dir = 'cropped_image'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0
cropped_images = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape


    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 for people in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes:
        if i < len(boxes): 
            x, y, w, h = boxes[i]
            person_crop = frame[y:y + h, x:x + w]
            person_filename = os.path.join(output_dir, f'person_{frame_count}_{i}.jpg')

            if not person_crop.size == 0:
                cv2.imwrite(person_filename, person_crop)
                cropped_images.append((person_filename, (x, y, w, h))) 

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

#   resizing images 
def resize_images(input_folder, output_folder, new_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(os.path.join(input_folder, filename)) as img:
                img = img.resize(new_size, Image.ANTIALIAS)
                img.save(os.path.join(output_folder, filename))

if __name__ == "__main__":
    input_folder = "cropped_image"
    output_dir ='final_cropped_resized_image'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_size = (128,256 )  
    
    resize_images(input_folder, output_dir, new_size)
    print("Images resized and saved to the output folder.")

# Extracting features using CNN

images_folder = 'final_cropped_resized_image'
input_shape = (128, 256, 3)

# Creating a list to store image data, labels, and filenames
images = []
labels = []
image_filenames = []  

for image_filename in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_filename)

    img = load_img(image_path, target_size=input_shape[:2])
    img_array = img_to_array(img)
    images.append(img_array)
    
    label = image_filename.split('_')[-1]  
    label = label.split('.')[0] 
    labels.append(int(label))

    image_filenames.append(image_filename)
images = np.array(images)
labels = np.array(labels)
images = images / 255.0

# Creating a simple CNN model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') 
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(images, labels, epochs=5, batch_size=32)

feature_extractor = Sequential(model.layers[:-2]) 
features= feature_extractor.predict(images)

np.save('image_filenames.npy', image_filenames)
print("Feature extraction and saving image filenames completed.")
np.save('features.npy', features)
features = np.load('features.npy')
image_filenames = np.load('image_filenames.npy')
print("Features shape:", features.shape)
print("Image filenames shape:", image_filenames.shape)

#  Coverting features.npy into csv format
num_frames, num_features = features.shape
reshaped_array = features.reshape(num_frames, -1)

with open('features.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(reshaped_array)

#   Visualising clusters using PCA and Kmeans and saving them in folder
non_zero_columns = np.any(features!= 0, axis=0)
selected_features = features[:, non_zero_columns]

num_clusters = 14
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(selected_features)

# Applying PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(selected_features)

plt.figure(figsize=(8, 6))
for cluster_id in range(num_clusters):
    cluster_data_points = reduced_features[cluster_labels == cluster_id]
    plt.scatter(cluster_data_points[:, 0], cluster_data_points[:, 1], label=f'Cluster {cluster_id}')

plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Creating output directory
output_dir = 'clustered_images'
os.makedirs(output_dir, exist_ok=True)
image_directory = "C:\\Person_Reid\\final_cropped_resized_image"
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.jpg') or filename.endswith('.png')]

# Creating directories for each cluster
for cluster_id in range(num_clusters):
    cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
    os.makedirs(cluster_dir, exist_ok=True)

# Moving images to respective cluster directories
for img_path, cluster_id in zip(image_paths, cluster_labels):
    cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
    img_filename = os.path.basename(img_path)
    new_img_path = os.path.join(cluster_dir, img_filename)
    shutil.copy(img_path, new_img_path)
