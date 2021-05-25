import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
api = KaggleApi()
api.authenticate()

api.dataset_download_file('kazanova/sentiment140', path='./')

# with zipfile.ZipFile('path/to/data.zip', 'r') as zipref:
#     zipref.extractall('target/path')