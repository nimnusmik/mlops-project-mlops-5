import os
import sys
import glob
import pickle

sys.path.append(
    os.path.dirname(
		    os.path.dirname(
				    os.path.dirname(
						    os.path.abspath(__file__)
						)
				)
		)
)

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from scripts.utils.utils import model_dir, calculate_hash, read_hash
from scripts.model.movie_predictor import MoviePredictor
from scripts.dataset.watch_log import WatchLogDataset, get_datasets
from scripts.dataset.data_loader import SimpleDataLoader
from scripts.evaluate.evaluate import evaluate
from scripts.postprocess.inference_to_db import write_db


def load_checkpoint(logger):
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
        
    all_models = sorted(glob.glob(models_path))

    if not all_models:
        logger.write(f"No model found in {target_dir}", print_error=True) 
        raise FileNotFoundError(f"No model found in {target_dir}")
    
    latest_model = all_models[-1]
    logger.write(f"Latest model found: {latest_model}") 

    if model_validation(latest_model, logger):
        with open(latest_model, "rb") as f:
            checkpoint = pickle.load(f)
        logger.write(f"Successfully loaded checkpoint from {latest_model}")
        return checkpoint
    else:
        logger.write("Hash mismatch: invalid file.", print_error=True)
        raise FileExistsError("Hash mismatch: invalid file.")


def init_model(checkpoint, logger):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    logger.write("Model initialized from checkpoint.")
    return model, scaler, label_encoder


def model_validation(model_path, logger):
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        logger.write(f"Model validation success for {model_path}")
        return True
    else:
        logger.write(f"Model validation failed for {model_path}: Hash mismatch.", print_error=True)
        return False
    

def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(
        data=[data],
        columns=columns
    )


def inference(model, scaler, label_encoder, data, logger, batch_size=1):
    logger.write("Starting inference process...")
    if data.size > 0:
        df = make_inference_df(data)
        print("Inference input df:")
        print(df)
        dataset = WatchLogDataset(df, scaler=scaler, label_encoder=label_encoder)
        logger.write("Inference data provided, creating WatchLogDataset from input.")
    else:
        _, _, dataset = get_datasets(scaler=scaler, label_encoder=label_encoder)
        logger.write("No inference data provided, using test dataset for inference.")

    print("Dataset size:", len(dataset))
    if len(dataset) == 0:
        print("Warning: Empty dataset after filtering unknown content_ids.")
        return []

    dataloader = SimpleDataLoader(
        dataset.features, dataset.labels, batch_size=batch_size, shuffle=False
    )
<<<<<<< Updated upstream
    _, predictions, accuracy = evaluate(model, dataloader)
    # print("Predictions:", predictions)
=======
    
    #_, predictions, accuracy = evaluate(model, dataloader)
    loss, predictions, accuracy = evaluate(model, dataloader)
    logger.write(f"Inference complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}") # <-- logger.write 사용
    # print(loss, predictions, accuracy)
>>>>>>> Stashed changes

    return [dataset.decode_content_id(idx) for idx in predictions]

 
def recommend_to_df(recommend):
    if not recommend:
        return pd.DataFrame(columns=["recommend_content_id"])

    if isinstance(recommend[0], (list, np.ndarray)):
        recommend = [item for sublist in recommend for item in sublist]

    return pd.DataFrame(
        data=recommend,
        columns=["recommend_content_id"]
    )