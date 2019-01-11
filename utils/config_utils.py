from argparse import Namespace

import yaml


def dict_to_args(arg_dict):
    return Namespace(**arg_dict)


def args_to_dict(name_space):
    return dict(name_space._get_kwargs())


def yaml_load_dict(file_path):
    f = open(file_path, 'r', encoding='utf-8')

    cont = f.read()

    arg_dicts = yaml.load(cont)

    return arg_dicts


def dict_to_yaml(fname, dicts):
    # dict = {}
    # for key, val in kwargs.items():
    #     dict['key'] = val
    with open(fname, "w") as f:
        yaml.dump_all(dicts.items(), f)


if __name__ == "__main__":
    """
    Test dict_to_args
    """

    x = yaml_load_dict("../configs/ptb.yaml")

    args = dict_to_args(x)

    dicts = args_to_dict(args)

    dict_to_yaml(fname=".test", dicts=dicts)
    print(args)
    print(dicts)
