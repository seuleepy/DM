# ML

myModule.py
사용자 정의 함수

preprocess.py
.npy 파일을 불러와서 .h5 파일로 변환

datacheck.py
변환한 .h5 파일 내용 확인

DataLoader.py
.h5 파일을 불러와서 train, val, test 파일로 split
torch.utils.data 의 Dataset을 상속받아 class를 정의 (train시에 이 파일의 class를 부름)
std scl 혹은 minmax scl를 사용하여 scaling
testset을 h5. 파일로 저장

Model.py
신경망 정보

train.py
학습
weight 정보를 weightFIle.pth 저장
epoch, loss, acc 정보를 history.csv 파일로 저장

acc_loss.py
history.csv 파일을 불러와 accuracy, loss 시각화

Eval.py
torch.utils.data.DataLoader를 사용 dataset option에 DataLoader.py의 TestDataset을 입력하여 test_loader를 정의
weightFIle.pth를 사용하여 testset을 평가
평가 내용을 prediction.csv 파일로 저장

analysis.py
prediction.csv 파일을 사용하여 roc_auc curve, DNN score, significance (DNN score에 대해 significance를 optimize한 결과)를 시각화

analysis_2.py
mediator 질량에 따른 분포를 확인하기 위함

cor.py
input feature의 correlation 확인

df_drop.py
correlation에서 불필요한 feature를 확인 후 input feature에서 삭제
.h5 파일로 저장

read_prediction.py
prediction.py 파일을 읽기 위함
