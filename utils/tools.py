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


class PostProcess(object):
    def __init__(self, sos, eos, tgt_vocab, src_vocab, src_pad, tgt_pad):
        self.sos = sos
        self.eos = eos
        self.tgt = tgt_vocab
        self.src = src_vocab
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def extract_single_source(self, source):
        process = []
        for tok in source:
            if tok == self.src_pad:
                pass
            else:
                process.append(self.src.itos[tok])
        return " ".join(process)

    def extract_single_target(self, target):
        process = []
        for tok in target:
            if tok == self.sos or tok == self.tgt_pad:
                pass
            elif tok == self.eos:
                break
            else:
                process.append(self.tgt.itos[tok])

        return " ".join(process)

    def fix_translate(self):
        pass
