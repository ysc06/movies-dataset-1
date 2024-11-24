import os
import requests
import gzip
import shutil
from gensim.models import KeyedVectors

def load_word2vec():
    # URL to the raw file on GitHub
    file_url = "https://github.com/ysc06/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz"

    # Local paths
    compressed_path = "./models/GoogleNews-vectors-negative300.bin.gz"
    model_path = "./models/GoogleNews-vectors-negative300.bin"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(compressed_path), exist_ok=True)

    # Download the file if it doesn't already exist
    if not os.path.exists(model_path):
        if not os.path.exists(compressed_path):
            print("Downloading GoogleNews-vectors-negative300.bin.gz...")
            response = requests.get(file_url, stream=True)
            with open(compressed_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download completed.")

        # Extract the .gz file
        print("Extracting GoogleNews-vectors-negative300.bin.gz...")
        with gzip.open(compressed_path, "rb") as f_in:
            with open(model_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction completed.")

    # Load the Word2Vec model
    print("Loading Word2Vec model...")
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")

    return word2vec_model

