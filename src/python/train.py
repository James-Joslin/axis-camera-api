import json
import util
from network_config import SqueezeNetConfig

if __name__ == "__main__":
    mobileNet_config = SqueezeNetConfig()
    util.set_seed(mobileNet_config.SEED)
    
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
    dataset_builder = util.datasetBuilder(
        secrets["PostProcessData"]["imagesTrain"],
        secrets["PostProcessData"]["annosTrain"]    
    )
    dataset_builder.build_datasets()
