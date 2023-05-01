import cv2
import os
import time
import requests
import cv2
# from mtcnn import MTCNN
import pandas as pd
import matplotlib.image as mpimg
import mtcnn
import tensorflow as tf
from tensorflow import keras
import numpy as np


def remane_file(dir_name,name,type,mini_range = 1):
    # Set the directory where the images are located
    directory = f"{dir_name}/{name}"

    # Set the starting number for the filenames
    count = mini_range
    file_name = []
    enrollment_ids = []
    if (type == 'img'):
        # Loop through all the files in the directory
        for filename in os.listdir(directory):
            # Check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                # Create the new filename by adding the count and the original extension
                new_filename = str(count) + os.path.splitext(filename)[1]
                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                # Increment the count for the next file
                count += 1

    elif(type == 'vdo'):
        directory = f'{dir_name}'
        # Loop through all the files in the directory
        for filename in os.listdir(directory):
            # Check if the file is an image
            if filename.endswith(".mp4"):
                # Create the new filename by adding the count and the original extension
                new_filename = "p" + str(count) + os.path.splitext(filename)[1]
                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                c = str(count)
                file_name.append(f'p{c}')
                enrollment_ids.append(f'{filename.split(".")[0]}')
                # Increment the count for the next file
                count += 1
        data = {
            'class_name' : file_name,
            'enrolment_no' : enrollment_ids
        }
        df = pd.DataFrame(data)
        df.to_csv('mapping.csv')

    else:
        print('try again')

def makedataset(person_name):
    # Define the video capture device
    cap = cv2.VideoCapture(0)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
 
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    # Define the output directory for the extracted frames
    output_dir = "train/" + person_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the total number of frames to extract
    total_frames = 90
    frame_count = 0

    while True:
        # Capture a frame from the video
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print('Error capturing frame')
            break

        # Display the frame in a window
        cv2.imshow('frame', frame)

        # Wait for a key press and check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save every 10th frame to the output directory
        if frame_count % 10 == 0 and frame_count < total_frames * 10:
            frame_file = os.path.join(output_dir, f'frame{frame_count // 10:03d}.jpg')
            cv2.imwrite(frame_file, frame)

        # Increment the frame count
        frame_count += 1

        # Stop capturing frames if we have extracted the desired number of frames
        if frame_count >= total_frames * 10:
            break

    # Release the video capture device and close the window
    cap.release()
    cv2.destroyAllWindows()

def makedataset_with_mp4(path_to_video,path_to_store,person_name,sub_folder='',frame = 90):
        # Define the video capture device
        # Path of the video file
    if sub_folder != '':
        video_path = f'{path_to_video}/{sub_folder}/{person_name}.mp4'
    else:
        video_path = f'{path_to_video}/{person_name}.mp4'

    if sub_folder != '':
        frame_folder = f"{path_to_store}"
    else:
        frame_folder = f"{path_to_store}/" + person_name

    # Path of the folder to save the frames

    # Create the folder if it does not exist
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) and total number of frames
    # fps = video.get(cv2.CAP_PROP_FPS)
    # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through all frames in the video
    for i in range(frame):
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the current frame
        ret, frame = video.read()

        # If the frame was read successfully, save it as an image
        if ret:
            frame_path = os.path.join(frame_folder, f'{i}.jpg')
            cv2.imwrite(frame_path, frame)

    # Release the video object
    video.release()

def makedataset_unknown():


    # Set the number of images to download
    num_images = 30

    # Create the photo folder if it doesn't exist
    if not os.path.exists("train/unknown"):
        os.makedirs("train/unknown")

    # Loop through the number of images and download them
    for i in range(num_images):
        # Get a random image URL
        response = requests.get("https://source.unsplash.com/random")

        
        # Get the file extension
        file_ext = response.headers.get("content-type").split("/")[-1]
        print()
        
        # # Save the image to the photo folder
        with open(f"train/unknown/frames_{i+1}.{file_ext}", "wb") as f:
            f.write(response.content)

        time.sleep(3)

def crop_image(inp_dir='train',out_dir='train2',name=''):

    # Set the directory where the images are located
    # input_directory = f"train/{name}"
    input_directory = f"{inp_dir}/{name}"

    # Set the directory where the cropped images will be saved
    # output_directory = f"train2/{name}"
    output_directory = f"{out_dir}/{name}"

    # Create the photo folder if it doesn't exist
    if not os.path.exists(f"{out_dir}/{name}"):
        os.makedirs(f"{out_dir}/{name}")

    # Create the MTCNN detector object
    detector = mtcnn.MTCNN()

    # Loop through all the files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
            # Read the image
            image = cv2.imread(os.path.join(input_directory, filename))
            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect faces in the image using MTCNN
            faces = detector.detect_faces(image)
            # Loop through each detected face
            for face in faces:
                # Get the coordinates of the bounding box around the face
                x, y, width, height = face["box"]
                # Crop the face from the image
                face_image = image[y:y+height, x:x+width]
                # Save the cropped face to the output directory
                cv2.imwrite(os.path.join(output_directory, f"cropped_{filename}"), face_image)

def list_dir(path):
    # specify the folder path
    # folder_path = 'train2/'
    folder_path = path
    # get a list of all the directories in the folder
    directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    # print(directories)
    return directories

def get_dataset_ready():

    # for video
    remane_file('train_video/','','vdo')

    # make data set from video
    folder_path = "train_video/"

    mp4_count = 0
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            mp4_count += 1
    
    path = 'sub_train'
    for i in range(mp4_count):
        i = i+1
        name = f"p{i}"
        makedataset_with_mp4(path_to_video='train_video/',path_to_store='sub_train/',person_name=name)

    # crop face from dataset and save it to final folder
    for i in range(mp4_count):
        i = i+1
        name = f"p{i}"
        crop_image(inp_dir='sub_train',out_dir='train',name=name)

    # makedataset_unknown()

def predict(img_path,new_model):

    print("////////////////////////////////////////////////////////////////////////")
    # img = mpimg.imread('2.jpg')
    img = mpimg.imread(img_path)

    face_detector = mtcnn.MTCNN()
    face_roi = face_detector.detect_faces(img)

    no_people = len(face_roi)
    print('Number of people in image is : ', no_people)
    total_people = []

    # print(total_people)    

    for i in range(no_people):

        x1, y1, width, height = face_roi[i]["box"]
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]

        resized_face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        # resized_face = face.resize((224, 224))
        input_image = np.array(resized_face)
        input_image = np.expand_dims(input_image, axis=0)
        print(resized_face.shape)
        # input_image = preprocess_input(input_image)

        prob_model = keras.Sequential ([
        new_model,
        tf.keras.layers.Softmax ()
        ])
        predictions = prob_model.predict(resized_face[None, ...])

        class_names = list_dir('train/')

        # assume `predictions` is a numpy array containing the predicted class indices
        predicted_class_indices = np.argmax(predictions, axis=1)

        # create a dictionary mapping the integer class indices to their string labels
        class_index_to_label = {i: class_names[i] for i in range(len(class_names))}

        # map the predicted class indices to their string labels
        predicted_labels = [class_index_to_label[i] for i in predicted_class_indices]

        print("predicted people : ", predicted_labels[0])
        total_people.append(predicted_labels[0])    


    print("////////////////////////////////////////////////////////////////////////")


    return total_people

# makedataset_unknown()
# makedataset_with_mp4("maurya")
# remane_file('maurya')
# makedataset(person_name)
# crop_image('bakul')
# list_dir('train/')

# get_dataset_ready()
