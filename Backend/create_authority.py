from getpass import getpass
from database import db
from utils.security import hash_password

print("===== CiviAI Authority Account Setup =====")

email = "authority@civiai.com"

existing_user = db.users.find_one({"email": email})

if existing_user:
    print("Authority email already exists!")
    print("No changes were made.")
else:
    password = getpass("Enter new authority password: ")
    confirm = getpass("Confirm authority password: ")

    if len(password) < 8:
        print("Password must be at least 8 characters.")

    elif password != confirm:
        print("Passwords do not match.")

    else:
        db.users.insert_one({
            "name": "CiviAI Administrator",
            "email": email,
            "password": hash_password(password),
            "role": "authority",
        })

        print("\nAuthority account created successfully!")
        print("Email: authority@civiai.com")
        print("Role: authority")