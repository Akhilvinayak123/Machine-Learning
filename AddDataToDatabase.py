import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendance-539ff-default-rtdb.firebaseio.com/"
})

ref = db.reference('Employees')

data = {
    "121212":
        {
            "name": "Ian Pulford",
            "major": "CEO",
            "last_attendance_time": "2023-10-26 00:54:34"

        },

    "321654":
        {
            "name": "Akhil Hariharasudhan",
            "major": "Robotics Engineer",
            "last_attendance_time": "2023-10-26 00:54:34"

        },

    "456512":
        {
            "name": "Baran Sahan",
            "major": "Head of Operations",
            "last_attendance_time": "2023-10-26 00:54:34"

        },

    "999191":
        {
            "name": "Casey Ashton-Smith",
            "major": "Head of systems",
            "last_attendance_time": "2023-10-26 00:54:34"

        }

}

for key, value in data.items():
    ref.child(key).set(value)