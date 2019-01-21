"""
nltk tokenizer for raw data.

"""
import argparse

from nltk.tokenize import word_tokenize


def post_token_process(line):
    line = line.strip()
    line = line.replace("(", "-LRB-")
    line = line.replace(")", "-RRB-")
    line = line.replace("[", "-LRB-")
    line = line.replace("]", "-RRB-")
    line = line.replace("{", "-LRB-")
    line = line.replace("}", "-RRB-")
    return line


def tokenizer(line, for_parse=False, is_lower=False):
    post = word_tokenize(line.strip())
    post_str = " ".join(post)
    if for_parse:
        post_str = post_token_process(post_str)
    if is_lower:
        post_str = post_str.lower()
    return post_str


def tokenizing(arr_list, for_parse=False, is_lower=False):
    """
    Args:
        arr_list: list<str>
        for_parse: bool
    :return:
    """
    return [tokenizer(line.strip(), for_parse, is_lower) for line in arr_list]


def write_docs(fname, docs):
    with open(fname, 'w') as f:
        for doc in docs:
            f.write(str(doc))
            f.write('\n')


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--raw_file', dest="raw_file", type=str, help='config_files')
    opt_parser.add_argument('--token_file', dest="token_file", type=str, help='config_files')
    opt_parser.add_argument('--for_parse', dest="is_parse", action="store_true", default=False)
    opt_parser.add_argument('--is_lower', dest="is_lower", action="store_true", default=False)
    opt = opt_parser.parse_args()
    raw_data_list = load_docs(opt.raw_file)
    process_list = tokenizing(raw_data_list, for_parse=opt.is_parse, is_lower=opt.is_lower)

    write_docs(
        fname=opt.token_file,
        docs=process_list
    )
