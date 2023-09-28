import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import bisect
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv').drop(columns=['SAMPLE_ID'])
test = pd.read_csv('data/test.csv').drop(columns=['SAMPLE_ID'])






# datetime 컬럼 처리
train['ATA'] = pd.to_datetime(train['ATA'])
test['ATA'] = pd.to_datetime(test['ATA'])
# datetime을 여러 파생 변수로 변환
## datetime을 학습하기 위해 년 ~ 분까지 데이터를 나눔.
for df in [train, test]:
    df['year'] = df['ATA'].dt.year
    df['month'] = df['ATA'].dt.month
    df['day'] = df['ATA'].dt.day
    df['hour'] = df['ATA'].dt.hour
    df['minute'] = df['ATA'].dt.minute
    df['weekday'] = df['ATA'].dt.weekday

# datetime 컬럼 제거
train.drop(columns='ATA', inplace=True)
test.drop(columns='ATA', inplace=True)

# Categorical 컬럼 인코딩
categorical_features = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']
encoders = {}

for feature in tqdm(categorical_features, desc="Encoding features"):
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature].astype(str))
    le_classes_set = set(le.classes_)
    test[feature] = test[feature].map(lambda s: '-1' if s not in le_classes_set else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '-1')
    le.classes_ = np.array(le_classes)
    test[feature] = le.transform(test[feature].astype(str))
    encoders[feature] = le

# 결측치 처리
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)








def train_and_evaluate(model, model_name, X_train, y_train):
	print(f'Model Tune for {model_name}.')
	model.fit(X_train, y_train)

	feature_importances = model.feature_importances_
	sorted_idx = feature_importances.argsort()

	plt.figure(figsize=(10, len(X_train.columns)))
	plt.title(f"Feature Importances ({model_name})")
	plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align='center')
	plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
	plt.xlabel('Importance')
	plt.show()

	return model, feature_importances


X_train = train.drop(columns='CI_HOUR')
y_train = train['CI_HOUR']

# Model Tune for LGBM
lgbm_model, lgbm_feature_importances = train_and_evaluate(lgb.LGBMRegressor(), 'LGBM', X_train, y_train)







threshold = 90 # Your Threshold
low_importance_features = X_train.columns[lgbm_feature_importances < threshold]

X_train_reduced = X_train.drop(columns=low_importance_features) # column을 날리는 거라 데이터 개수에 영향 x
# print(X_train_reduced.shape, X_train.shape)

X_test_reduced = test.drop(columns=low_importance_features)

lgbm = lgb.LGBMRegressor(max_depth=-1, num_leaves=250, learning_rate=0.4)



# 5-Fold 설정
kf = KFold(n_splits=40, shuffle=True, random_state=42)

# 각 fold의 모델로부터의 예측을 저장할 리스트와 MAE 점수 리스트
ensemble_predictions = []
MAE = []

for train_idx, val_idx in tqdm(kf.split(X_train_reduced), total=40, desc="Processing folds"):
	X_t, X_val = X_train_reduced.iloc[train_idx], X_train_reduced.iloc[val_idx]
	y_t, y_val = y_train[train_idx], y_train[val_idx]

	# 두 모델 모두 학습
	lgbm.fit(X_t, y_t)

	# 각 모델로부터 Validation set에 대한 예측을 평균내어 앙상블 예측 생성
	val_pred = lgbm.predict(X_val)

	# Validation set에 대한 대회 평가 산식 계산 후 저장
	MAE.append(mean_absolute_error(y_val, val_pred))

	# test 데이터셋에 대한 예측 수행 후 저장
	lgbm_pred = lgbm.predict(X_test_reduced)
	lgbm_pred = np.where(lgbm_pred < 0, 0, lgbm_pred)

	ensemble_predictions.append(lgbm_pred)

# K-fold 모든 예측의 평균을 계산하여 fold별 모델들의 앙상블 예측 생성
final_predictions = np.mean(ensemble_predictions, axis=0)

# 각 fold에서의 Validation Metric Score와 전체 평균 Validation Metric Score출력
print("Validation : NMAE scores for each fold:", MAE)
print("Validation : NMAE:", np.mean(MAE))




"""

# 정규화 -> LGBM 적용 pipe 생성
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler

set_config(display = 'diagram')
pipe_lgbm = make_pipeline(StandardScaler(),
                          LGBMRegressor(n_estimate = 400, randomstate=1, metric = 'mse'))
pipe_lgbm.fit(X_train_reduced, y_train)

print(pipe_lgbm.get_params().keys())


# 그리드 서치
from sklearn.model_selection import GridSearchCV
param_grid = { 'lgbmregressor__max_depth':[3,5,8],
               'lgbmregressor__learning_rate' : [0.1,0.05,0.03,0.01],
              'lgbmregressor__min_child_samples' : [1,10,20,30],
              'lgbmregressor__min_child_weight' : [1,3,5]}

gs = GridSearchCV(estimator = pipe_lgbm,
                  scoring = 'accuracy',
                  cv = 5,  # 5겹 교차검증
                  param_grid = param_grid,
                  refit = True,  # 훈련후 바로 적용
                  return_train_score = True # 훈련 성능 리턴
                  )
gs.fit(X_train, y_train)
print(f'최적의 하이퍼파라미터 세트:{gs.best_params_}')


"""





submit = pd.read_csv('data/sample_submission.csv')
submit['CI_HOUR'] = final_predictions
submit.to_csv('results/baseline_submit.csv', index=False)


