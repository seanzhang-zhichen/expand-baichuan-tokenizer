import jsonlines
import sentencepiece as spm



def train_sp(train_file, domain_sp_model_name):
    max_sentence_length = 200000
    pad_id = 3
    model_type = "BPE"
    vocab_size = 8000


    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=domain_sp_model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=max_sentence_length,
        pad_id=pad_id,
        model_type=model_type,
        vocab_size=vocab_size,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
        num_threads=16
    )

def test_sp(model_file):
    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    # encode: text => id
    print(sp.encode_as_pieces('法院审查期间，双方均认为应当按照瑞典法律来理解《合同》中的仲裁条款。斯万斯克公司认为争议解决条款的中文意思是“如发生任何争议，应适用瑞典法律并在瑞典通过快速仲裁解决。”而常力蜂业公司则认为上述条款的中文意思是“为瑞典法律管辖下的争议在瑞典进行快速仲裁解决。'))
    print(sp.encode_as_ids('法院审查期间，双方均认为应当按照瑞典法律来理解《合同》中的仲裁条款。斯万斯克公司认为争议解决条款的中文意思是“如发生任何争议，应适用瑞典法律并在瑞典通过快速仲裁解决。”而常力蜂业公司则认为上述条款的中文意思是“为瑞典法律管辖下的争议在瑞典进行快速仲裁解决。'))



    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    print(sp.encode_as_ids('this is a test'))


if __name__ == '__main__':
    train_file = "../data/emample.txt"
    domain_sp_model_name = "../model/domain_sp"
    model_file = domain_sp_model_name + ".model"
    train_sp(train_file, domain_sp_model_name)
    test_sp(model_file)
