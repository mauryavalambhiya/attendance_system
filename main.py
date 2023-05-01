import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL
from PIL import Image
import os
import os. path
import pandas as pd
import collectdata as cds
import mtcnn
import datetime
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="attendance"
)

mycursor = mydb.cursor()

def list_dirs(path):

    folders = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    return folders

def convert_list_to_id(std_list):
    print("student list tobe converted   :  ",std_list)
    df = pd.read_csv('mapping.csv')
    new_list = []

    for i in std_list:
        ids = (df['enrolment_no'][df['class_name'] == str(i)])
        try:
            print('ids ...  ', list(ids)[0])
            new_list.append(list(ids)[0])
        except:
            continue
    
    print("new list   :  ",new_list)
    return new_list

def fill_attendance(path_to_video='test_video_photo/',sub_folder='ce_c/',name='test',path_to_store='test/ce_c',new_model='',subject = 'OS',faculty = 'NJS',sem = 3):

    frame = 30
    cds.makedataset_with_mp4(sub_folder=sub_folder,path_to_video=path_to_video,person_name=name,path_to_store=path_to_store,frame=frame)
    # cds.makedataset_with_mp4(sub_folder='ce_c/',path_to_video='test_video_photo/',person_name='test',path_to_store='test_video_photo/ce_c')
    input_directory = path_to_store
    detector = mtcnn.MTCNN()
    main_set = {}
    main_set = set(main_set)
    cds.remane_file(path_to_store,'','img',200)
    cds.remane_file(path_to_store,'','img',1)
    for i in range(1,frame+1):
        # Check if the file is an image
        # if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
        #     # Read the image
        #     image = cv2.imread(os.path.join(input_directory, filename))
        try:
            redicted_list = cds.predict(f'{path_to_store}/{i}.jpg',new_model=new_model)

            redicted_set = set(redicted_list)
            main_set.update(redicted_set)
            # print('main_set  .. ', main_set,)
        except:
            redicted_list = cds.predict(f'{path_to_store}/{i}.jpeg',new_model=new_model)
            redicted_set = set(redicted_list)
            main_set.update(redicted_set)

    
    attended_student = list(main_set)
    # print('attended_student :   ..........   ',attended_student)
    attended_student = convert_list_to_id(attended_student)
    # for i in attended_student:
    #     print('attended_student  .. ', i)

    # print(attended_student)

    student_list =attended_student
    my_string = sub_folder

    division = my_string.rstrip('/').lstrip('/')
    # subject = 'OS'
    # faculty = 'NJS'
    today = datetime.date.today()
    formatted_date = today.strftime('%Y-%m-%d')
    sem = 3

    status = insert_data(student_list,division,subject,faculty,formatted_date,sem)

    return status

def insert_data(student_list,division,subject,faculty,formatted_date,sem):

    # data = [(1, 'v1'), (2, 'v2'), (3, 'v3'), (4, 'v4'), (5, 'v5'), (6, 'v6')]
    data = []

    for i in student_list:
        add_tuple = (i, subject,faculty,formatted_date, 1,division,sem)
        data.insert(0, add_tuple)

    print(data)
    
    # not_present = list(set(total_student).difference(set(student_list)))

    # for i in not_present:
    #     add_tuple = (i, subject,faculty,formatted_date,0,division)
    #     data.insert(0, add_tuple)

    sql = 'INSERT INTO attendance_sys (enrollment_no, subject, faculty, formatted_date, is_present, division,sem) VALUES (%s, %s, %s, %s, %s, %s,%s)'
    mycursor.executemany(sql, data)
    mydb.commit()

    return True

new_model = tf.keras.models.load_model('saved_models/v_2_2_13_04_23.h5')
# new_model.summary()

# names1 = list(cds.predict('test_photos/1.jpg',new_model))
# names2 = list(cds.predict('test_photos/2.jpg',new_model))
# names3 = list(cds.predict('test_photos/3.jpg',new_model))
# print(names1)
# print(names2)
# print(names3)

path_to_video='test_video_photo/'
sub_folder='ce_c/'
name='test'
path_to_store=f'test/{sub_folder}'

my_string = sub_folder

division = my_string.rstrip('/').lstrip('/')
subject = 'OS'
faculty = 'NJS'
today = datetime.date.today()
formatted_date = today.strftime('%d-%m-%Y')
sem = 3
total_student = []

student_list = fill_attendance(path_to_video=path_to_video,sub_folder=sub_folder,name=name,path_to_store=path_to_store,new_model=new_model,subject = 'OS',faculty = 'NJS',sem = 3)
# student_list = ['21SOECE11147','21SOCCE11006']
# status = insert_data(student_list,division,subject,faculty,formatted_date,sem)

if (student_list):
    print('Attendance is done....')
else:
    print('Try again...')