from __future__ import absolute_import

import os
import sys

from metrics.vae_metrics import VaeEvaluator
from struct_self.dataset import Dataset
from struct_self.dataset import to_example
from utils.vae_utils import get_eval_dir
from utils.vae_utils import load_model


# sys.path.append(".")


def test_vae(main_args, model_args, input_mode=0):
    model = load_model(main_args, model_args, check_dir=False)
    out_dir = get_eval_dir(main_args=main_args, model_args=model_args, mode="Test")
    model.eval()
    if not os.path.exists(out_dir):
        sys.exit(-1)
    if model_args.model_select.startswith("Origin"):
        model_args.eval_bs = 20 if model_args.eval_bs < 20 else model_args.eval_bs
    evaluator = VaeEvaluator(
        model=model,
        out_dir=out_dir,
        eval_batch_size=model_args.eval_bs,
        train_batch_size=main_args.batch_size
    )
    train_exam = Dataset.from_bin_file(main_args.train_file).examples

    para_eval_dir = "/home/user_data/baoy/projects/seq2seq_parser/data/quora-mh/unsupervised"
    para_eval_list = ["dev.para.txt"]
    # ["dev.para.txt", "test.para.txt"]

    if input_mode == 0:
        print("========dev reconstructor========")
        test_set = Dataset.from_bin_file(main_args.dev_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="dev")
        print("finish")
        print("========test reconstructor=======")
        test_set = Dataset.from_bin_file(main_args.test_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="test")
        print("finish")
        print("========generating samples=======")
        evaluator.evaluate_generation(corpus_examples=train_exam, sample_size=len(test_set.examples), eval_desc="gen")
        print("finish")
    elif input_mode == 1:
        print("========generating samples=======")
        test_exam = Dataset.from_bin_file(main_args.test_file).examples
        evaluator.evaluate_generation(corpus_examples=train_exam, sample_size=len(test_exam), eval_desc="gen")
        print("finish")
    elif input_mode == 2:
        print("========generating paraphrase========")
        evaluator.evaluate_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 3:
        print("========supervised generation========")
        # evaluator.evaluate_control()
        evaluator.evaluate_control(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 4:
        trans_eval_list = ["trans.length.txt", "trans.random.txt"]
        print("========style transfer========")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=trans_eval_list, eval_desc="unmatch")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=para_eval_list, eval_desc="match")
        print("finish")
    elif input_mode == 5:
        print("========random syntax select========")
        evaluator.evaluate_pure_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    else:
        raw = input("raw sent: ")
        while not raw.startswith("EXIT"):
            e = to_example(raw)
            words = model.predict(e)
            print("origin:", " ".join(words[0][0][0]))
            to_ref = input("ref syn : ")
            while not to_ref.startswith("NEXT"):
                syn_ref = to_example(to_ref)
                ret = model.eval_adv(e, syn_ref)
                if not model_args.model_select == "OriginVAE":
                    print("ref syntax: ", " ".join(ret['ref syn'][0][0][0]))
                    print("ori syntax: ", " ".join(ret['ori syn'][0][0][0]))
                print("switch result: ", " ".join(ret['res'][0][0][0]))
                to_ref = input("ref syn: ")
            raw = input("input : ")
