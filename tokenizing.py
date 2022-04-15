from transformers import AutoTokenizer


TYPE = {"ORG": "단체", "PER": "사람", "DAT": "날짜", "LOC": "위치", "POH": "기타", "NOH": "수량"}

def tokenizing_data(train_dataset, tokenizer_name="klue/roberta-large", MODE = "default"):
    tokenizer = get_tokenizer(tokenizer_name, MODE)

    token_dataset = tokenized_dataset(train_dataset, tokenizer=tokenizer)

    return tokenizer, token_dataset

def get_tokenizer(tokenizer_name, MODE="default"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if MODE=="token":
        """
        tokenizer.add_special_tokens({'additional_special_tokens': ['[SUB;ORG]', '[/SUB;ORG]',
                                                                '[SUB;PER]', '[/SUB;PER]',
                                                                '[OBJ;PER]', '[/OBJ;PER]',
                                                                '[OBJ;LOC]', '[/OBJ;LOC]',
                                                                '[OBJ;DAT]', '[/OBJ;DAT]',
                                                                '[OBJ;ORG]', '[/OBJ;ORG]',
                                                                '[OBJ;POH]', '[/OBJ;POH]',
                                                                '[OBJ;NOH]', '[/OBJ;NOH]',
                                                                ]})
        """
        tokenizer.add_special_tokens({'additional_special_tokens':['[SUB]','[/SUB]','[OBJ]', '[/OBJ]']})
    elif MODE=="cv" or MODE=="add_sptok":
        tokenizer.add_special_tokens({"additional_special_tokens": ['[TP]', '[/TP]']})

    return tokenizer

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    # tokenizer.__call__
    concat_entity = []

    for e01, e02, t01, t02 in zip(
            dataset["subject_entity"],
            dataset["object_entity"],
            dataset["subject_type"],
            dataset["object_type"],
    ):
        temp = ""
        # temp = f"이 문장에서 @{e01}@과 #{e02}#은 어떤 관계일까?"  #
        temp = f"이 문장에서 [{e02}]은 [{e01}]의 [{TYPE[t02]}]이다."  # 현재 최고점 temp
        concat_entity.append(temp)
    """
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = e01 + '와 ' + e02 +'의 관계를 구하시오.'
        temp1 = f'*{e01}[{e01_type}]* 와  + *{e02}[{e02_type}]* 의 관계를 구하시오.'
        temp2 = f'이 문장에서 *{e01}*과 ^{e02}^은 어떤 관계일까?'  # multi 방식 사용
        temp3 = f"이 문장에서 [{e02}]은 [{TYPE[e01_type]}]인 [{e01}]의 [{TYPE[e02_type]}]이다."
        concat_entity.append(temp)
    """

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences
