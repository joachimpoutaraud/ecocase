import os
import sys
import contextlib
import numpy as np
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


def prepare_segments(roi, chunk_duration=3.0):
    # Calculate number of samples per chunk
    total_samples = int(chunk_duration * roi['rate'])

    try:
        # Extract ROI audio
        ndarray = roi['roi']
        num_samples = len(ndarray)
        base_tmin = roi.get('tmin', 0.0)

        chunks = []
        chunk_idx = 0

        for i in range(0, num_samples, total_samples):
            chunk = ndarray[i:i + total_samples]
            actual_samples = len(chunk)

            # Pad the last chunk if it's shorter than total_samples
            if actual_samples < total_samples:
                padded = np.zeros(total_samples, dtype=ndarray.dtype)
                padded[:actual_samples] = chunk
                chunk = padded

            # Calculate chunk-specific tmin/tmax
            chunk_tmin = base_tmin + (chunk_idx * chunk_duration)
            chunk_tmax = chunk_tmin + chunk_duration

            chunks.append({
                "path": roi['path'],
                "rate": roi['rate'],
                "roi": ndarray,
                "chunk": chunk,
                "idx": chunk_idx,
                "tmin": roi.get('tmin'),
                "tmax": roi.get('tmax'),
                "tmin_chunk": chunk_tmin,
                "tmax_chunk": chunk_tmax,
                "fmin": roi.get('fmin'),
                "fmax": roi.get('fmax'),
                "audio_filter_chain": roi.get('audio_filter_chain'),
            })

            chunk_idx += 1

    except Exception as e:
        print(f"Error processing ROI {roi.get('path')}: {e}")
        chunks = []

    return chunks


def analyze_segments(args):
    # Get the arguments
    seg, extract, predict, min_confidence = args

    with suppress_output():
        birdnet_analyzer = Analyzer()

    birdnet_sr = 48000  # BirdNET sampling rate
    chunk = seg["chunk"]
    rate = seg['rate']

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
        **{k: seg[k] for k in ["path", "rate", "roi", "chunk", "idx", "tmin", "tmax", "tmin_chunk", "tmax_chunk", "fmin", "fmax", "audio_filter_chain"]},
        "embeddings": embeddings if extract else None,
        "prediction": prediction,
        "confidence": confidence
    }

    return result