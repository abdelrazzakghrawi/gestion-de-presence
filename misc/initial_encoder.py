import cv2  # Importation de la bibliothèque OpenCV pour le traitement d'images et de vidéos
import pickle  # Importation du module pickle pour la sérialisation et la désérialisation des objets Python
import face_recognition  # Importation de la bibliothèque face_recognition pour la reconnaissance faciale
import os  # Importation du module os pour interagir avec le système d'exploitation
import firebase_admin  # Importation de la bibliothèque firebase_admin pour interagir avec Firebase
from firebase_admin import credentials  # Importation du module credentials de firebase_admin
from firebase_admin import db  # Importation du module db de firebase_admin pour Firebase Realtime Database
from firebase_admin import storage  # Importation du module storage de firebase_admin pour Firebase Storage


# Charger les informations d'identification Firebase depuis le fichier JSON
cred = credentials.Certificate('C:/Users/MY PC/Desktop/projet_tuto/gestion-de-presence/serviceAccountKey.json')
# Initialiser l'application Firebase avec les informations d'identification chargées et spécifier l'URL de la base de données et le bucket de stockage
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://gestion-dabsence-default-rtdb.firebaseio.com/",  # URL de la base de données Firebase Realtime Database
    'storageBucket': "gestion-dabsence.appspot.com"  # Nom du bucket de stockage Firebase Storage
})

folderPath = "static/Files/Images"  # Chemin du répertoire contenant les images des étudiants
# Liste des chemins d'accès aux fichiers d'images dans le dossier
imgPathList = os.listdir(folderPath)  # Utilisation de la fonction os.listdir() pour obtenir la liste des fichiers dans le dossier
print(imgPathList)  # Affiche la liste des chemins d'accès aux fichiers d'images dans le dossier



imgList = []  # Liste vide pour stocker les images des étudiants
studentIDs = []  # Liste vide pour stocker les identifiants des étudiants

# Parcourir la liste des chemins d'accès aux fichiers d'images
for path in imgPathList:
    # Charger chaque image et la stocker dans imgList
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  # Charger l'image à partir du chemin d'accès et l'ajouter à imgList

    # Extraire l'identifiant de l'étudiant à partir du nom du fichier
    studentIDs.append(os.path.splitext(path)[0])  # Extraire l'identifiant de l'étudiant en supprimant l'extension du fichier et l'ajouter à studentIDs

    # Télécharger l'image vers le stockage Firebase
    fileName = f"{folderPath}/{path}"  # Chemin complet du fichier
    bucket = storage.bucket()  # Récupérer le bucket de stockage Firebase
    blob = bucket.blob(fileName)  # Créer un objet blob pour le fichier dans le bucket
    blob.upload_from_filename(fileName)  # Télécharger le fichier vers le stockage Firebase


print(studentIDs)  # Afficher la liste des identifiants des étudiants



def findEncodings(images):
    encodeList = []  # Initialisation de la liste des encodages faciaux

    # Parcourir les images pour calculer les encodages faciaux
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir l'image de BGR (OpenCV) à RGB (face_recognition)
        encode = face_recognition.face_encodings(img)[0]  # Calculer l'encodage facial de l'imag
        encodeList.append(encode)  # Ajouter l'encodage facial à la liste des encodages
    return encodeList  # Retourner la liste des encodages faciaux


print("Encodage commencé")  # Afficher un message indiquant le début du processus d'encodage
# Appeler la fonction pour trouver les encodages faciaux pour les images de la liste
encodeListKnown = findEncodings(imgList)  # Appeler la fonction findEncodings avec la liste des images en entrée pour obtenir les encodages faciaux correspondants

# Regrouper les encodages avec les identifiants correspondants
encodeListKnownWithIds = [encodeListKnown, studentIDs]  # Créer une liste regroupant les encodages faciaux et les identifiants des étudiants

# Enregistrer les encodages dans un fichier pickle
file = open("EncodeFile.p", "wb")  # Ouvrir un fichier en mode écriture binaire
pickle.dump(encodeListKnownWithIds, file)  # Sérialiser les données (encodages et identifiants) et les écrire dans le fichier
file.close()  # Fermer le fichier

# Afficher un message indiquant la fin de l'encodage
print("Encodage terminé")  # Afficher un message indiquant la fin du processus d'encodage
