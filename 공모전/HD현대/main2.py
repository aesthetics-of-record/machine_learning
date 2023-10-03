import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import bisect
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


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

# lgbm = lgb.LGBMRegressor(max_depth=-1, num_leaves=250, learning_rate=0.4)











# 정규화 -> LGBM 적용 pipe 생성
from lightgbm import LGBMRegressor
from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV


set_config(display = 'diagram')


# 그리드 서치
from sklearn.model_selection import GridSearchCV


param_grid = { 'max_depth':[-1,3,5,8],
			   'n_estimators': [200, 500, 1000, 2000],
               'learning_rate' : [0.5, 0.4, 0.3, 0.2, 0.1,0.05,0.03,0.01],
			   'num_leaves': [100, 150, 200, 250, 300, 500, 700],
			   }

print("작동?")

gs = GridSearchCV(estimator = LGBMRegressor(),
                  scoring = 'neg_mean_squared_error',
                  cv = 5,  # 5겹 교차검증
                  param_grid = param_grid,
                  refit = True,  # 훈련후 바로 적용
                  return_train_score = True # 훈련 성능 리턴
                  )

gs.fit(X_train_reduced, y_train)

print(f'최적의 하이퍼파라미터 스코어:{gs.best_score_}')
print(f'최적의 하이퍼파라미터 세트:{gs.best_params_}')