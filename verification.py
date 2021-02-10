from deepface import DeepFace


img1 = "images/dis.jpg"
img2 = "images/v.jpg"

models = ["Facenet","VGG-Face"]

res = DeepFace.verify(img1,img2,model_name=models[0])

# saved path is: /home/milad4274/.deepface/weights/vgg_face_weights.h5


print("Is verified: ", res["verified"])
print(res)


# obj = DeepFace.analyze(img1,['emotion'])
# print(obj)
