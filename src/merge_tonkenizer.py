# -*- coding: utf-8 -*-

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tokenization_baichuan import BaichuanTokenizer
import sentencepiece as spm


def load_user_vocab(vocab_file):
    # Read jieba vocab and sort by freq
    with open(vocab_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        word_freqs = [line.strip().split() for line in lines]
        word_freqs.sort(key=lambda x: int(x[1]), reverse=True)
    return word_freqs


def merge_vocab(src_tokenizer_dir, domain_sp_model_file, output_sp_dir, output_hf_dir, domain_vocab_file=None):

    # load
    src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_dir, trust_remote_code=True)
    domain_sp_model = spm.SentencePieceProcessor()
    domain_sp_model.Load(domain_sp_model_file)

    src_tokenizer_spm = sp_pb2_model.ModelProto()
    src_tokenizer_spm.ParseFromString(src_tokenizer.sp_model.serialized_model_proto())
    domain_spm = sp_pb2_model.ModelProto()
    domain_spm.ParseFromString(domain_sp_model.serialized_model_proto())


    print("原始词表大小： ", len(src_tokenizer))
    print("用自己的数据训练出来的词表大小", len(domain_sp_model))
    
    # 原始词表集合
    src_tokenizer_spm_tokens_set = set(p.piece for p in src_tokenizer_spm.pieces)
    print("原始词表 example: ", list(src_tokenizer_spm_tokens_set)[:10])


    # 拓展词表
    added_set = set()
    for p in domain_spm.pieces:
        piece = p.piece
        if piece not in src_tokenizer_spm_tokens_set:
            # print('新增词：', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            src_tokenizer_spm.pieces.append(new_p)
            added_set.add(piece)
    
    print(f"加了sp训练出来的词表后，新词表大小: {len(src_tokenizer_spm.pieces)}")


    # 加载自定义的词表
    if domain_vocab_file:
        word_freqs = load_user_vocab(domain_vocab_file)
        domain_vocab_set = set([i[0] for i in word_freqs if i])
        print('自定义的词表 example:', list(domain_vocab_set)[:10])
        print('自定义的词表大小:', len(domain_vocab_set))
        for p in domain_vocab_set:
            piece = p
            if piece not in src_tokenizer_spm_tokens_set and piece not in added_set:
                # print('新增词：', piece)
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                src_tokenizer_spm.pieces.append(new_p)
        print(f"加了自定义的词表后，新词表大小: {len(src_tokenizer_spm.pieces)}")

    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/new_tokenizer.model', 'wb') as f:
        f.write(src_tokenizer_spm.SerializeToString())
    tokenizer = BaichuanTokenizer(vocab_file=output_sp_dir + '/new_tokenizer.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"new tokenizer has been saved to {output_hf_dir}")


def test_new_tokenizer(src_tokenizer_dir, new_tokenizer_dir):
    print("-------"*10 + "test" + "-------"*10)
    # Test
    src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_dir, trust_remote_code=True)
    new_tokenizer = BaichuanTokenizer.from_pretrained(new_tokenizer_dir)

    print('原始词表大小:', len(src_tokenizer), '新词表大小:', len(new_tokenizer))
    print()
    text = '''
2018 年 11 月 13 日，原被告双方通过网络在线签订了《个人贷款担保服务合同》，合同约定，被告委托原告为其与银行的个人贷款提供担保等服务，原告向被告收取担保费，担保费费率按照每月0.71%

计收。合同还约定，若被告任意一期还款逾期达到 85 个自然日或者

出借人宣布借款提前到期但被告未能及时偿还时，原告履行全额代偿责任。原告已承担担保责任的，被告未能及时偿还代偿款项，被告应向原告支付代偿资金占用费，代偿资金占用费=全额代偿资金本金× 2%×（逾期天数/30）。

此后，被告与银行签署了《个人贷款借款合同》，合同约定银行向被告提供贷款人民币 250000 元，贷款期限为 36 月，年利率为 9.5%， 还款方式为等额本息还款，还约定若被告逾期未足额还款则产生逾期利息、罚息等费用。

合同履行过程中，被告未按照约定及时向银行偿还借款，原告共向银行支付代偿款 172890.94 元（含剩余未还本金、逾期利息、罚息等），但被告未按照合同约定向原告偿还代偿款及相关费用，已经构成违约。

截至 2020 年 9 月 4 日，被告已逾期还款 172 天，为维护原告的合法权益，原告特根据相关法律规定，向人民法院提起民事诉讼，望判如所请。
    '''
    print(f"Tokenized by src tokenizer:{src_tokenizer.tokenize(text)}")
    print(f"Tokenized by src tokenizer length: {len(src_tokenizer.tokenize(text))} ")
    print(f"Tokenized by new tokenizer:{new_tokenizer.tokenize(text)}")
    print(f"Tokenized by new tokenizer length: {len(new_tokenizer.tokenize(text))} ")
    


if __name__ == '__main__':
    output_sp_dir = '../model/merged_tokenizer_sp' # sentencepiece 词表格式保存路径
    output_hf_dir = '../model/merged_tokenizer_hf' # huggingface 词表格式保存路径
    src_tokenizer_dir = "/home/llm_models/Baichuan-13B-Chat" # 换成你自己的模型文件夹
    domain_sp_model_file = "../model/domain_sp.model"
    domain_vocab_file = "../data/法律诉讼.txt"
    merge_vocab(src_tokenizer_dir, domain_sp_model_file, output_sp_dir, output_hf_dir, domain_vocab_file)
    test_new_tokenizer(src_tokenizer_dir, output_hf_dir)
