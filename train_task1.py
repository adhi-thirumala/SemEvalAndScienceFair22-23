#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    ret = []
    for _, i in df.iterrows():
        ret += [{'text': ' '.join(i['postText']) + ' - ' + i['targetTitle'] + ' ' + ' '.join(i['targetParagraphs']), 'uuid': i['uuid']}]
    
    return pd.DataFrame(ret)

    #check cuda
def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0
train_df = pd.read_json('train.jsonl', lines=True)  
train_df