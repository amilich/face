import face_recognition

image = face_recognition.load_image_file("obama.jpg")
face_locations = face_recognition.face_locations(image)#, model="cnn")
print(face_locations)
