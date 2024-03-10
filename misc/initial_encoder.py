import cv2
import pickle
import face_recognition
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Charger les informations d'identification Firebase
cred = credentials.Certificate('C:/Users/MY PC/Desktop/projet_tuto/gestion-de-presence/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://gestion-dabsence-default-rtdb.firebaseio.com/",
    'storageBucket': "gestion-dabsence.appspot.com"
})

# Chemin du dossier contenant les images des étudiants
folderPath = "static/Files/Images"
imgPathList = os.listdir(folderPath)
print(imgPathList)

# Initialiser des listes pour stocker les images et les identifiants des étudiants
imgList = []
studentIDs = []

# Parcourir la liste des fichiers d'images
for path in imgPathList:
    # Charger chaque image et la stocker dans imgList
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    # Extraire l'identifiant de l'étudiant à partir du nom du fichier
    studentIDs.append(os.path.splitext(path)[0])

    # Télécharger l'image vers le stockage Firebase
    fileName = f"{folderPath}/{path}"
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

# Afficher les identifiants des étudiants
print(studentIDs)


def findEncodings(images):
    encodeList = []

    # Parcourir les images pour calculer les encodages faciaux
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]

        encodeList.append(encode)

    return encodeList


print("Encodage commencé")

# Appeler la fonction pour trouver les encodages faciaux
encodeListKnown = findEncodings(imgList)

# Regrouper les encodages avec les identifiants correspondants
encodeListKnownWithIds = [encodeListKnown, studentIDs]

# Enregistrer les encodages dans un fichier pickle
file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()

# Afficher un message indiquant la fin de l'encodage
print("Encodage terminé")
