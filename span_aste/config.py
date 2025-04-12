EMOTION_LABELS = None
NUM_CLASSES = None

def init_emotion_labels(dataset_name):
    global EMOTION_LABELS
    global NUM_CLASSES
    EMOTION_LABELS = get_emotion_labels(dataset_name)
    NUM_CLASSES = len(EMOTION_LABELS)

def get_emotion_labels(data):
    if data == "facebook":
        return {
            "NEUTRAL": 0,
            "JOY": 1,
            "SURPRISE": 2,
            "ANGER": 3,
            "SADNESS": 4,
            "FEAR": 5,
            "DISGUST": 6,
            "INVALID": 7
        }
    if data == "final_df6":
        return {
            "NEUTRAL": 0,
            "JOY": 1,
            "SADNESS": 2,
            "DISGUST": 3,
            "ANGER": 4,
            "FEAR": 5,
            "SURPRISE": 6,
            "INVALID": 7
        }
    if data == "final_df6B":
        return {
            "JOY": 0,
            "SADNESS": 1,
            "DISGUST": 2,
            "ANGER": 3,
            "FEAR": 4,
            "SURPRISE": 5,
            "INVALID": 6
        }
    if data == "final_df6C":
        return {
            "NEGATIVE": 0,
            "POSTIVE": 1,
            "INVALID": 2
        }
    raise ValueError(f"[ERROR] Unknown dataset: {data}")
