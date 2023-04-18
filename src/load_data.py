import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data,read_params




def load_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df = df.replace(np.nan,'')
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path,sep=",",index=False)
    return df






if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_data(config_path = parsed_args.config)

