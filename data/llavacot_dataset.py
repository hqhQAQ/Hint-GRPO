import os
import re
import json
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from math_verify import LatexExtractionConfig, parse, verify

filter_indexes_image_size = [201, 334, 349, 556, 732, 741, 777, 946, 991, 1005, 1129, 1187, 1201, 1448, 1467, 1780, 1886, 1935, 1952, 2010, 2253, 2409, 2690, 2780, 2918, 2942, 3023, 3026, 3296, 3584, 3717, 3740, 3788, 3833, 3887, 3945, 4592, 4696, 4907, 4990, 5385, 5443, 5653, 5735, 5758, 5839, 5995, 6018, 6030, 6190, 6375, 6385, 6680, 6799, 6939, 7169, 7670, 7770, 7786, 7826, 7830]  # Image size too small or large

def get_filter_indexes_llavacot(sample_len_file, filter_length):
    with open(sample_len_file, 'rb') as file:
        sample_lens = pickle.load(file)
    sample_lens = np.array(sample_lens)
    filter_indexes = np.nonzero(sample_lens >= filter_length)[0]
    filter_indexes = list(filter_indexes)

    return filter_indexes

def judge_alpha(sol):
    return any(char.isupper() for char in sol)

def judge_numeric(sol):
    sol_parserd = parse(sol.strip(), extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    if len(sol_parserd) == 0:
        pattern = r'-?\d+(?:\.\d+)?'
        numbers = re.findall(pattern, sol)
        return True if numbers else False
    else:
        return True

def delete_options(question):
    options = re.findall(r'[A-D]\. .*', question)
    if options:
        return re.sub(r'[A-D]\..*$', '', question, flags=re.DOTALL)
    else:
        return question
    return

def get_first_capital(sol):
    res, cnt = None, 0
    for char in sol:
        if char in 'ABCD':
            res = char
            cnt += 1
    if cnt > 1:
        return None
    else:
        return res

def extract_options(question):
    parts = re.split(r'\s*([A-D]\.)\s*', question)
    result = {}
    for i in range(1, len(parts), 2):
        key = parts[i].strip('.')
        value = parts[i+1].strip() if i+1 < len(parts) else ""
        result[key] = value

    return result

def split_steps(reasoning_text):
    pattern = r'Step \d+'
    steps = re.split(pattern, reasoning_text)
    steps = [step.strip(' :\n') for step in steps]
    return steps[1:]

class LLaVACoTDataset(Dataset):
    def __init__(self, data_root, split, json_file):
        self.data_root = data_root
        self.all_samples = []
        if split != 'train':
            return
        sample_idx = 0
        with open(json_file, 'r') as file:
            for line in file:
                json_object = json.loads(line)
                conversations = json_object['conversations']
                if len(conversations) == 0:
                    continue
                # Processing, select samples for geometry: geoqa+, CLEVR_v1.0
                selected_dirs = ['geoqa+', 'CLEVR_v1.0']
                assert len(conversations) % 2 == 0
                num_samples = len(conversations) // 2
                for s_idx in range(num_samples):
                    if json_object['image'].split('/')[0] not in selected_dirs:
                        continue
                    sample = {}
                    sample['id'] = json_object['id']
                    sample['index'] = sample_idx
                    sample['image'] = json_object['image']
                    sample['question'] = conversations[2 * s_idx]['value']
                    answer = conversations[2 * s_idx + 1]['value']
                    # Get summary
                    pattern = r'<SUMMARY>(.*?)</SUMMARY>'
                    match = re.search(pattern, answer, re.DOTALL)
                    sample['summary'] = match.group(1).lstrip(' \n').rstrip(' \n')
                    # Get reasoning
                    pattern = r'<REASONING>(.*?)</REASONING>'
                    match = re.search(pattern, answer, re.DOTALL)
                    sample['reasoning'] = match.group(1).lstrip(' \n').rstrip(' \n')
                    # Get conclusion
                    pattern = r'<CONCLUSION>(.*?)</CONCLUSION>'
                    match = re.search(pattern, answer, re.DOTALL)
                    sample['solution'] = match.group(1)
                    # Get reasoning split steps
                    sample['reasoning_steps'] = split_steps(sample['reasoning'])
                    
                    first_option = get_first_capital(sample['solution'])
                    if first_option is not None: # Filter option questions
                        sample['solution'] = extract_options(sample['question'])[first_option]
                    if judge_numeric(sample['solution']) is False:  # Filter answers without numbers
                        continue
                    sample['question'] = delete_options(sample['question'])
                    self.all_samples.append(sample)
                    sample_idx += 1

    def __getitem__(self, index):
        res = {}
        
        sample = self.all_samples[index]
        img_path = os.path.join(self.data_root, sample['image'])
        img = Image.open(img_path)

        res['index'] = sample['index']
        res['image_path'] = [img_path]
        res['image'] = [img]
        res['prompt'] = {'role': 'user', 'content': sample['question']}
        res['solution'] = sample['solution']
        res['summary'] = sample['summary']
        res['reasoning'] = sample['reasoning']
        res['reasoning_steps'] = sample['reasoning_steps']

        return res
    
    def __len__(self):
        return len(self.all_samples)