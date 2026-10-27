
import pandas as pd
from pathlib import Path

complaints = [
    # LOW
    ("Garbage", "A small amount of litter is beside a public bench.", 1, 0, 0, 0, "Low"),
    ("Streetlight", "One streetlight is not working on a quiet residential lane.", 1, 0, 0, 0, "Low"),
    ("WaterLeakage", "A tap at a public park is dripping slowly.", 1, 0, 0, 0, "Low"),
    ("RoadDamage", "A minor crack is visible on the edge of a quiet road.", 1, 0, 0, 0, "Low"),
    ("Garbage", "A small waste bag was left near a collection point.", 1, 0, 0, 0, "Low"),
    ("Streetlight", "A lamp near a walking path is dim but the path remains visible.", 1, 0, 0, 0, "Low"),
    ("Drainage", "Water is draining slowly from a small roadside channel.", 1, 0, 0, 0, "Low"),
    ("RoadDamage", "A shallow surface crack needs routine maintenance.", 1, 0, 0, 0, "Low"),
    ("Garbage", "Some dry leaves have accumulated around a public bin.", 1, 0, 0, 0, "Low"),
    ("WaterLeakage", "A minor leak is reported at an outdoor water point.", 1, 0, 0, 0, "Low"),

    # MEDIUM
    ("Garbage", "Several overflowing bags are creating an unpleasant smell.", 2, 0, 0, 0, "Medium"),
    ("WaterLeakage", "Water is leaking continuously from a roadside pipe.", 2, 0, 0, 0, "Medium"),
    ("Streetlight", "Multiple streetlights are not working along a residential street.", 2, 0, 0, 0, "Medium"),
    ("Drainage", "Rainwater is collecting near houses because a drain is partly blocked.", 2, 0, 0, 0, "Medium"),
    ("RoadDamage", "A pothole is forming on a road with moderate traffic.", 2, 0, 0, 0, "Medium"),
    ("Garbage", "Waste has remained uncollected for several days near shops.", 2, 0, 0, 0, "Medium"),
    ("WaterLeakage", "A broken public tap is wasting water throughout the day.", 2, 0, 0, 0, "Medium"),
    ("Drainage", "A blocked drain is causing water to pool on a footpath.", 2, 0, 0, 0, "Medium"),
    ("RoadDamage", "A damaged section of pavement makes walking uncomfortable.", 2, 0, 0, 0, "Medium"),
    ("Streetlight", "Poor lighting is affecting visibility on a neighborhood road.", 2, 0, 0, 0, "Medium"),

    # HIGH
    ("Pothole", "A deep pothole on a busy road makes vehicles swerve.", 3, 0, 1, 1, "High"),
    ("Drainage", "A blocked drain is flooding the main pedestrian walkway.", 3, 0, 0, 1, "High"),
    ("WaterLeakage", "A major water pipe leak is flooding a busy street.", 3, 0, 1, 1, "High"),
    ("ElectricalHazard", "Exposed electrical wiring is hanging beside a public walkway.", 2, 0, 0, 1, "High"),
    ("Garbage", "A large pile of waste is blocking part of a busy road.", 3, 0, 1, 0, "High"),
    ("Pothole", "Several large potholes are creating a serious road hazard.", 3, 0, 1, 1, "High"),
    ("Drainage", "Sewage is overflowing onto a crowded pedestrian area.", 3, 0, 0, 1, "High"),
    ("Streetlight", "All lights are out at a busy junction with poor visibility.", 3, 0, 1, 1, "High"),
    ("RoadDamage", "A damaged road section is forcing buses to change lanes.", 3, 0, 1, 1, "High"),
    ("WaterLeakage", "A burst pipe is making the road slippery for vehicles.", 3, 0, 1, 1, "High"),

    # CRITICAL
    ("ElectricalHazard", "A fallen live power line is sparking in a crowded street.", 3, 0, 1, 1, "Critical"),
    ("Flooding", "Severe flooding has trapped people inside nearby buildings.", 3, 0, 1, 1, "Critical"),
    ("Drainage", "A person has fallen into an uncovered deep drain.", 3, 0, 0, 1, "Critical"),
    ("RoadDamage", "A collapsed road has created an immediate danger to traffic.", 3, 0, 1, 1, "Critical"),
    ("ElectricalHazard", "An electrical pole has fallen across a public road.", 3, 0, 1, 1, "Critical"),
    ("Flooding", "Rapid floodwater is entering homes and blocking evacuation routes.", 3, 0, 1, 1, "Critical"),
    ("WaterLeakage", "A major pipe rupture has undermined the road and caused a collapse.", 3, 0, 1, 1, "Critical"),
    ("ElectricalHazard", "Live wires are touching standing water near a crowded area.", 3, 0, 0, 1, "Critical"),
    ("RoadDamage", "A large road collapse has left a dangerous open gap.", 3, 0, 1, 1, "Critical"),
    ("Drainage", "Severe sewage flooding is creating an immediate public safety hazard.", 3, 0, 0, 1, "Critical"),
]

columns = [
    "category",
    "description",
    "detections_count",
    "near_school_or_hospital",
    "blocks_traffic",
    "safety_risk",
    "priority",
]

df = pd.DataFrame(complaints, columns=columns)

output_path = Path(__file__).resolve().parent / "civiai_priority_training_dataset_v2.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Total complaints:", len(df))
print("\nExamples per priority:")
print(df["priority"].value_counts().sort_index())
print("\nSaved to:", output_path)
