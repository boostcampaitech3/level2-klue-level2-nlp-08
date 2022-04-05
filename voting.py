from load_data import *
import pandas as pd
import torch, random
import torch.nn.functional as F
from collections import Counter
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import gc

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


# 결과 csv 경로 예시 : f'./prediction/ensemble/{num}.csv' (1 <= num <= args.ensemble_num)

def main(args):
    ## predict answer
    pred_answer_list = []
    output_prob_list = []
    test_id = []
    for i in range(1,args.ensemble_num +1):
        SUB_CSV = args.submission_dir + str(i) + '.csv'  # csv dir.
        test_ensemble = pd.read_csv(SUB_CSV)
        temp_pred_answer, temp_output_prob = list(test_ensemble["pred_label"]), list(test_ensemble["probs"])
        pred_answer_list.append(temp_pred_answer)
        output_prob_list.append(temp_output_prob)
    test_id = list(range(len(pred_answer_list[0])))

    if args.ensemble_type=="hard":
        # ver1 : hard voting
        pred_answer = []
        output_prob = output_prob_list[0]
        
        for idx in range(len(pred_answer_list[0])):
            c = Counter([pred_answer_list[n][idx] for n in range(0,args.ensemble_num)])
            pred_answer.append(c.most_common(1)[0][0])
        pred_answer = num_to_label(pred_answer)
    else:
        # ver2 : soft voting
        output_prob = []
        pred_answer = []
        for j in range(len(output_prob_list[0])):
            prob = []
            for k in range(30):
                c = 0
                for i in range(args.ensemble_num):
                    c += output_prob_list[i][j][k]
                prob.append(c/args.ensemble_num)
            output_prob.append(prob)
            pred_answer.append(np.argmax(prob))

        pred_answer = num_to_label(pred_answer)

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    output.to_csv(
        f"{args.submission_dir}{args.submission_name}.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("---- Finish! ----")


if __name__ == "__main__":
    gc.collect()
    parser = argparse.ArgumentParser()

    # model dir
    # default : "./prediction/ensemble/{num}.csv" 경로에 num번째 앙상블할 csv파일 존재 / "./prediction/ensemble/result.csv" 경로에 최종 csv 저장
    parser.add_argument("--submission_dir", type=str, default=f"./prediction/ensemble/")
    parser.add_argument("--submission_name", type=str, default="result")
    parser.add_argument('--ensemble_num', type=int, default=3)
    parser.add_argument("--ensemble_type", type=str, default="hard")
    args = parser.parse_args()
    print(args)
    main(args)
