import argparse
from easydict import EasyDict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def options() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="./")
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_kfold', type=str2bool, default=False)
    parser.add_argument('--kfold_splits', type=int, default=4)
    parser.add_argument('--model_name', type=str, default="deit_small_patch16_224")
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_len', type=int, default=590)
    parser.add_argument('--epochs', type=int, default=1)

    # =======================
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--voting', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()

    crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}

    disease = {
        '1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
        '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
              'b8': '다량원소결핍 (K)'},
        '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
        '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
        '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
        '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}

    risk = {'1': '초기', '2': '중기', '3': '말기'}

    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                    '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

    args.num_features = len(csv_features)
    label_option = EasyDict({"crop" : crop,
                             "disease": disease,
                             "risk" : risk,
                             "csv_features" : csv_features})

    return args, label_option
