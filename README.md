Statoil-C-CORE-Iceberg-Classifier-Challenge
===========================================

-	Site: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

Description
-----------

Goal
----

-	이미지 데이터를 통해 Iceberg인지 Ship인지 여부 예측
-	Test Dataset에 대해 확률값으로 예측

Submission File Format
----------------------

![Submission_File_Format](./Image/Submission_File_Format.png)

Dataset
-------

-	Training Set: 1604개의 데이터 / 3개의 Column으로 구성
-	Test Set: 8424개의 데이터 / 2개의 Column으로 구성

Model
-----

-	4개의 Convolution Layer와 3개의 Fully Connected Layer로 구성되어 있으며 Output Layer로 Sigmoid 함수를 써서 확률 값으로 표현
-	Loss Function으로 CrossEntropy 사용 / Optimizer로 Adam을 사용

Result
------

### Hyper Parameter

-	Epoch: 50
-	Batch Size : 12

![Result](./Image/Result.png)

Discussion
----------

-	Convolution Neural Network로 Binary Classification을 구현
-	끝난 Competition으로 정확한 결과값을 확인할 수 없었다.
