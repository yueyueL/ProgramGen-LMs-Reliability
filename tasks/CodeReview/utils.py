


def update_tokenizer(tokenizer, load_extra_ids=True,add_lang_ids=False):
    adds = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<keep>", "<add>", "<del>", "<start>", "<end>"]
    # adds = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    adds = [tok for tok in adds if tok not in tokenizer.get_vocab()]
    if adds:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": adds}
        )
        
    if load_extra_ids:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<extra_id_{}>".format(i) for i in range(99, -1, -1)
                ]
            }
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<e{}>".format(i) for i in range(99, -1, -1)
                ]
            }
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["<msg>"]})
    
    langs = [
        "<en>",
        "<python>",
        "<java>",
        "<javascript>",
        "<ruby>",
        "<php>",
        "<go>",
        "<c>",
        "<c_sharp>",
        "<c_plus_plus>",
    ]
    if add_lang_ids:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": langs
            }
        )
        
    tokenizer.special_dict = {
        f"<e{i}>" : tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }
    # confusing api...
    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]
    
    return tokenizer            