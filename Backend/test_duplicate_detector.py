
from duplicate_detector import find_similar_complaints

existing_complaints = [
    {
        "complaint_id": "CIV-TEST-001",
        "category": "Pothole",
        "description": "A large pothole near the main bus stop",
        "status": "REPORTED",
    },
    {
        "complaint_id": "CIV-TEST-002",
        "category": "Garbage",
        "description": "Garbage has not been collected for several days",
        "status": "REPORTED",
    },
]

new_description = "There is a big road hole beside the bus stop"

matches = find_similar_complaints(
    description=new_description,
    category="Pothole",
    existing_complaints=existing_complaints,
)

print("Possible duplicate complaints:")

if not matches:
    print("No matches found above the similarity threshold.")
else:
    for match in matches:
        print(match)
