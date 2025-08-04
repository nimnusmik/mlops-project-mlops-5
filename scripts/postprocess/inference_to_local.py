import os
import datetime
import pandas as pd

from scripts.utils.utils import project_path


def save_inference_to_local(df, model_name="model", save_dir="data/processed"):
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{model_name}_{timestamp}.csv"
    save_path = os.path.join(project_path(), save_dir, filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Inference result saved to {save_path}")
    return save_path