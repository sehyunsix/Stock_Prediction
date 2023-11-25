import argparse
from transformers import HfArgumentParser
from models.TimeSeries.TCN.TCN import TCNModel
from  dataclasses import dataclass

@dataclass
class TCNargumenets(HfArgumentParser):
        input_size : int = 5
        output_size : int = 20
        channels_num : int = 80
        channels_level: int =3
        kernel_size : int =7
        dropout : float = 0.25
# 인자값을 받을 수 있는 인스턴스 생성
 # 입력받을 인자값 등록
# parser.add_argument('--target', required=True, help='어느 것을 요구하냐')
# parser.add_argument('--env', required=False, default='dev', help='실행환경은 뭐냐')
# 입력받은 인자값을 args에 저장 (type: namespace)
# args = parser.parse_args()

#

# args1 = parser.parse_args()
# args = parser2.parse_args()

def main():
    parser = HfArgumentParser(TCNargumenets,description="Train")
    args1 = parser.parse_args()
    print(args1)
    args2 =parser.parse_args()
    print(args2)
    model =TCNModel(args1.__dict__)
    print(model)

if __name__ == '__main__':
    main()