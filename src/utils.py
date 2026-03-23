import string
import re

# Characters: A-Z + 0-9
CHAR_SET = string.ascii_uppercase + string.digits
char_to_num = {c: i+1 for i, c in enumerate(CHAR_SET)}
num_to_char = {i+1: c for i, c in enumerate(CHAR_SET)}

def encode_label(text, max_len=10):
    label = [char_to_num[c] for c in text if c in char_to_num]
    label += [0] * (max_len - len(label))
    return label

def decode_label(pred):
    return ''.join([num_to_char.get(i, '') for i in pred if i != 0])

def valid_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return re.match(pattern, text) is not None