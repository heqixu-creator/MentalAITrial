#!/usr/bin/env python3
"""
Generate a sample dataset for testing the pipeline when the real dataset is not available.
Creates data/dataset.csv with columns 'text' and 'label' (0/1).
Sample texts are illustrative only; for real research use a proper dataset (see README).
"""

import csv
import random
from pathlib import Path

# Example phrases suggestive of depression (label 1) vs neutral/positive (label 0)
# These are for pipeline testing only.
DEPRESSION_PHRASES = [
    "I feel empty and hopeless most days",
    "Nothing seems to matter anymore",
    "I can't get out of bed",
    "I have no energy for anything",
    "I've been really down lately",
    "I don't see the point in trying",
    "I feel so alone even when people are around",
    "I've lost interest in everything I used to enjoy",
    "I can't stop crying",
    "I feel like a burden to everyone",
    "Sleep is the only escape",
    "I don't want to be here anymore",
    "Everything feels pointless",
    "I'm exhausted all the time",
    "I've been isolating myself",
    "I feel numb and disconnected",
    "I don't deserve to be happy",
    "My anxiety and sadness are overwhelming",
    "I just want to disappear",
    "I've been in a dark place",
]

NEUTRAL_PHRASES = [
    "Had a good day at work today",
    "Looking forward to the weekend",
    "The weather is nice this morning",
    "Just finished reading a great book",
    "Meeting friends for dinner later",
    "Trying out a new recipe tonight",
    "Got some exercise in today",
    "Planning a trip for next month",
    "My cat is being silly again",
    "Coffee and a quiet morning",
    "Finished the project I was working on",
    "Going to the gym after work",
    "Watching a movie with family",
    "Learned something new today",
    "Had a productive day",
    "Taking a walk in the park",
    "Catching up on some sleep",
    "Cleaning the house and feeling accomplished",
    "Video call with old friends",
    "Ordered takeout for dinner",
]


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "dataset.csv"

    random.seed(42)
    rows = []
    for _ in range(450):
        rows.append((random.choice(DEPRESSION_PHRASES), 1))
    for _ in range(450):
        rows.append((random.choice(NEUTRAL_PHRASES), 0))
    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
