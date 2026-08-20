import gdown
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_folder(folder_url: str, output_dir: Path) -> Path:
    """Download an entire public/shared Google Drive folder using gdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info(f"Downloading URL: {folder_url} -> {output_dir}")
    gdown.download_folder(
        url=folder_url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
    )
    return output_dir

def download_and_save_data(dataset_name: str, output_dir: Path) -> Path:
    """Download the raw dataset and save it to the output directory."""
    if dataset_name == "Berka":
        folder_url = f"https://drive.google.com/drive/folders/1LA23XSQTin7p6oWtxg7GL7_Qm_sSJ4sn"
        logger.info(f"Downloading the transaction table of theBerka dataset from {folder_url} -> {output_dir}")
        download_folder(folder_url, output_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in the supported datasets.\
            Please implement your own dataset download function.")
    return output_dir