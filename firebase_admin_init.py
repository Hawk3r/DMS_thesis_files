import firebase_admin
from firebase_admin import credentials, firestore

# Replace with your downloaded private key path from Firebase Console
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
