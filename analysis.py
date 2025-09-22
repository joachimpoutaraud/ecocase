import os
import sys
import contextlib
from operator import itemgetter

import librosa

from birdnetlib.analyzer import Analyzer

# Suppress stdout/stderr from birdnetlib
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def analyze_segment(args):
    # Get the arguments
    seg, rate, extract, predict, min_confidence = args

    with suppress_output():
        birdnet_analyzer = Analyzer()

    birdnet_sr = 48000  # BirdNET sampling rate
    chunk = seg["chunk"]

    if rate != birdnet_sr:
        chunk = librosa.resample(chunk, orig_sr=rate, target_sr=birdnet_sr)

    embeddings = None
    prediction = "Unknown"
    confidence = 0.0

    if extract:
        embeddings = birdnet_analyzer._return_embeddings([chunk])[0].tolist()

    if predict:
        predictions = birdnet_analyzer.predict(chunk)[0]
        label_scores = dict(zip(birdnet_analyzer.labels, predictions))
        sorted_labels = sorted(label_scores.items(), key=itemgetter(1), reverse=True)
        filtered = [item for item in sorted_labels if item[1] >= min_confidence]

        if filtered:
            prediction, confidence = filtered[0]

    # Create result dict by unpacking seg and updating only necessary keys
    result = {
        **{k: seg[k] for k in ["path", "roi", "chunk", "idx", "tmin", "tmax", "fmin", "fmax"]},
        "embeddings": embeddings if extract else None,
        "prediction": prediction,
        "confidence": confidence
    }

    return result