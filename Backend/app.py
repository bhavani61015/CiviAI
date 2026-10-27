
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import uuid4
from pathlib import Path
import sys
import tempfile
import joblib
import pandas as pd
from bson import ObjectId

from database import db
from models.user import User
from utils.security import hash_password, verify_password
from duplicate_detector import find_similar_complaints


# --------------------------------------------------
# XGBoost PRIORITY MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "priority_model_v2.joblib"
)

saved_priority_model = joblib.load(MODEL_PATH)
priority_model = saved_priority_model["model"]
priority_label_to_id = saved_priority_model["label_to_id"]

priority_id_to_label = {
    value: key for key, value in priority_label_to_id.items()
}


def predict_complaint_priority(complaint_data: dict) -> str:
    description = complaint_data.get("description") or ""
    category = complaint_data.get("category") or ""
    detections = complaint_data.get("detections") or []

    text = description.lower()
    category_lower = category.lower()

    safety_keywords = (
        "danger", "unsafe", "injury", "fallen wire",
        "sparking", "flood", "collapsed", "open drain",
    )

    traffic_keywords = (
        "blocking traffic", "blocks traffic", "busy road",
        "vehicles to swerve", "traffic disruption",
    )

    features = pd.DataFrame([{
        "category": category,
        "description": description,
        "detections_count": len(detections),
        "near_school_or_hospital": int(
            "school" in text or "hospital" in text
        ),
        "blocks_traffic": int(
            any(word in text for word in traffic_keywords)
        ),
        "safety_risk": int(
            any(word in text for word in safety_keywords)
            or category_lower in {
                "electricalhazard", "flooding"
            }
        ),
    }])

    predicted_id = int(priority_model.predict(features)[0])
    return priority_id_to_label[predicted_id]


# --------------------------------------------------
# YOLO DETECTOR
# --------------------------------------------------

YOLO_DIR = (
    Path(__file__).resolve().parent.parent
    / "ai"
    / "yolo"
)

sys.path.insert(0, str(YOLO_DIR))

from detector import CivicIssueDetector

civic_detector = CivicIssueDetector()


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SECRET_KEY = "civiai-development-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ComplaintCreate(BaseModel):
    category: str
    description: str
    image_file_name: str
    location: Optional[str] = None
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    detections: Optional[List[Dict[str, Any]]] = None


class ComplaintStatusUpdate(BaseModel):
    complaint_id: str
    status: str


class ComplaintFeedbackCreate(BaseModel):
    complaint_id: str
    rating: int = Field(ge=1, le=5)
    resolved_properly: bool
    message: str = Field(min_length=1, max_length=1000)



# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CiviAI API",
    description=(
        "Intelligent Civic Issue Detection, "
        "Prioritization and Management System"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:56255",
        "http://127.0.0.1:56255",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# AUTHENTICATED USER / ROLE VERIFICATION
# --------------------------------------------------

bearer_scheme = HTTPBearer()


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        user_id = payload.get("sub")
        if not user_id or not ObjectId.is_valid(user_id):
            raise ValueError("Invalid user ID")
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise ValueError("User no longer exists")
        return user
    except (JWTError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired login") from exc


def require_citizen(user=Depends(current_user)):
    if user.get("role", "").lower() != "citizen":
        raise HTTPException(status_code=403, detail="Citizen access required")
    return user


def require_authority(user=Depends(current_user)):
    if user.get("role", "").lower() != "authority":
        raise HTTPException(status_code=403, detail="Authority access required")
    return user


# --------------------------------------------------
# HOME AND HEALTH
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "CiviAI Backend is running!",
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/db-test")
def db_test():
    db.command("ping")

    return {
        "message": "MongoDB connected successfully!",
        "status": "success",
    }


# --------------------------------------------------
# USER REGISTRATION
# --------------------------------------------------

@app.post("/register")
def register_user(user: User):
    email = user.email.strip().lower()

    existing_user = db.users.find_one({"email": email})

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    user_data = user.model_dump()

    user_data["email"] = email
    user_data["password"] = hash_password(user.password)

    # Public registration creates citizen accounts only.
    user_data["role"] = "citizen"

    result = db.users.insert_one(user_data)

    return {
        "message": "User registered successfully!",
        "user_id": str(result.inserted_id),
        "role": "citizen",
    }

# --------------------------------------------------
# USER LOGIN
# --------------------------------------------------

@app.post("/login")
def login_user(
    email: str = Form(...),
    password: str = Form(...),
):
    user = db.users.find_one({
    "email": email.strip().lower()
})

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    if not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    token_data = {
        "sub": str(user["_id"]),
        "email": user["email"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc)
        + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }

    access_token = jwt.encode(
        token_data,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return {
        "message": "Login successful!",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }


# --------------------------------------------------
# YOLO IMAGE DETECTION
# --------------------------------------------------

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    allowed_extensions = (
        ".jpg", ".jpeg", ".png", ".webp", ".gif"
    )

    original_filename = file.filename or ""
    suffix = Path(original_filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG, PNG, WEBP and GIF "
                "images are allowed"
            ),
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Image file is empty",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(image_data)
            temp_path = Path(temp_file.name)

        detections = civic_detector.detect(str(temp_path))

        return {
            "message": "Image analyzed successfully",
            "filename": original_filename,
            "content_type": file.content_type,
            "size": len(image_data),
            "detection_count": len(detections),
            "detections": detections,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image detection failed. Please check "
                "the image and model."
            ),
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# --------------------------------------------------
# SUBMIT A COMPLAINT
# --------------------------------------------------

@app.post("/complaints")
def create_complaint(
    complaint: ComplaintCreate,
    user=Depends(require_citizen),
):
    complaint_data = complaint.model_dump()

    existing_complaints = list(
        db.complaints.find(
            {},
            {
                "_id": 0,
                "complaint_id": 1,
                "category": 1,
                "description": 1,
                "status": 1,
            },
        ).limit(500)
    )

    duplicate_matches = find_similar_complaints(
        description=complaint_data.get("description", ""),
        category=complaint_data.get("category", ""),
        existing_complaints=existing_complaints,
        threshold=0.70,
        max_results=5,
    )

    predicted_priority = predict_complaint_priority(
        complaint_data
    )

    complaint_id = (
        f"CIV-{datetime.now().strftime('%Y%m%d')}-"
        f"{uuid4().hex[:6].upper()}"
    )

    complaint_data.update({
        "complaint_id": complaint_id,
        "citizen_id": str(user["_id"]),
        "status": "REPORTED",
        "priority": predicted_priority,
        "created_at": datetime.now(timezone.utc),
        "possible_duplicates": duplicate_matches,
    })

    result = db.complaints.insert_one(complaint_data)

    return {
        "message": "Complaint submitted successfully",
        "complaint_id": complaint_id,
        "citizen_id": str(user["_id"]),
        "status": "REPORTED",
        "priority": predicted_priority,
        "database_id": str(result.inserted_id),
        "possible_duplicates": duplicate_matches,
        "duplicate_count": len(duplicate_matches),
    }


# --------------------------------------------------
# FETCH ALL COMPLAINTS
# --------------------------------------------------

@app.get("/complaints")
def get_complaints(user=Depends(current_user)):
    # Citizens see only their own complaints; authorities see all.
    query = {} if user.get("role", "").lower() == "authority" else {
        "citizen_id": str(user["_id"])
    }
    complaints = list(
        db.complaints.find(
            query,
            {"_id": 0},
        ).sort("created_at", -1)
    )

    return {
        "message": "Complaints fetched successfully",
        "count": len(complaints),
        "complaints": complaints,
    }


# --------------------------------------------------
# UPDATE COMPLAINT STATUS
# --------------------------------------------------

@app.patch("/complaints/status")
def update_complaint_status(
    update: ComplaintStatusUpdate,
    user=Depends(require_authority),
):
    allowed_statuses = {
        "REPORTED",
        "IN_PROGRESS",
        "RESOLVED",
    }

    new_status = update.status.strip().upper()

    if new_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. Use REPORTED, "
                "IN_PROGRESS, or RESOLVED."
            ),
        )

    result = db.complaints.update_one(
        {"complaint_id": update.complaint_id},
        {"$set": {"status": new_status}},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Complaint not found.",
        )

    return {
        "message": "Complaint status updated successfully",
        "complaint_id": update.complaint_id,
        "status": new_status,
    }


# --------------------------------------------------
# FEEDBACK ON RESOLVED COMPLAINTS ONLY
# --------------------------------------------------

@app.post("/feedback", status_code=201)
def submit_feedback(
    feedback: ComplaintFeedbackCreate,
    user=Depends(require_citizen),
):
    complaint = db.complaints.find_one({
        "complaint_id": feedback.complaint_id,
        "citizen_id": str(user["_id"]),
    })
    if complaint is None:
        raise HTTPException(status_code=404, detail="Your complaint was not found")
    if complaint.get("status") != "RESOLVED":
        raise HTTPException(
            status_code=400,
            detail="Feedback is allowed only after the complaint is resolved",
        )
    if db.feedback.find_one({"complaint_id": feedback.complaint_id}):
        raise HTTPException(status_code=409, detail="Feedback already submitted")

    data = feedback.model_dump()
    data["citizen_id"] = str(user["_id"])
    data["created_at"] = datetime.now(timezone.utc)
    result = db.feedback.insert_one(data)
    return {
        "message": "Complaint feedback submitted successfully",
        "feedback_id": str(result.inserted_id),
    }


@app.get("/feedback")
def get_feedback(user=Depends(current_user)):
    if user.get("role", "").lower() == "authority":
        query = {}
    else:
        query = {"citizen_id": str(user["_id"])}
    records = list(db.feedback.find(query, {"_id": 0}).sort("created_at", -1))
    return {"feedback": records, "count": len(records)}
