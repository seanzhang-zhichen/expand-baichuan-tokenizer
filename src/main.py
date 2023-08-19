import os
from train_sp import train_sp
from merge_tonkenizer import merge_vocab
from merge_tonkenizer import test_new_tokenizer


def main():
    model_dir = "../model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    train_file = "../data/emample.txt" # 训练文件路径
    domain_sp_model_name = "../model/domain_sp" # 训练好的词表保存dir
    train_sp(train_file, domain_sp_model_name)


    output_sp_dir = '../model/merged_tokenizer_sp' # sentencepiece 词表格式保存路径
    output_hf_dir = '../model/merged_tokenizer_hf' # huggingface 词表格式保存路径
    
    src_tokenizer_dir = "/home/dibai/zhangzc/MedicalGPT/zhifa-llm-based-baichuan-13b-chat" # 原始词表路径
    domain_sp_model_file = domain_sp_model_name + ".model" # 训练好的词表保存的路径
    domain_vocab_file = "../data/法律诉讼.txt" # 自定义的词表路径
    merge_vocab(src_tokenizer_dir, domain_sp_model_file, output_sp_dir, output_hf_dir, domain_vocab_file)
    test_new_tokenizer(src_tokenizer_dir, output_hf_dir)



if __name__ == "__main__":
    main()