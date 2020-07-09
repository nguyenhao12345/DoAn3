# import nhung thu vien cho chuong trinh can 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# load data va replace NaN thanh gia tri 0
train_data = pd.read_csv("train.csv").fillna(0)
test_data = pd.read_csv("test.csv").fillna(0)

#lay cot thuoc tinh Survived bo vao y_train
y_train = train_data["Survived"]

# train_data xoa cot Survived
train_data.drop(labels="Survived", axis=1, inplace=True)

# gop train_data voi test_data bo vao full_data
full_data = train_data.append(test_data)
drop_columns = ["Name", "SibSp", "Ticket", "Cabin", "Parch", "Embarked", "PassengerId"]

# sau do bo cac thuoc tinh khong can phan loai
full_data.drop(labels=drop_columns, axis=1, inplace=True)

# tao sex thanh nhung cot rieng voi moi gia tri
full_data = pd.get_dummies(full_data, columns=["Sex"])
full_data = pd.get_dummies(full_data, columns=["Pclass"])

# sau khi chuan hoa du lieu xong bo chia du lieu ra thanh 2 phan bo vao X_train va X_test
X_train = full_data.values[0:891]
X_test = full_data.values[891:]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

state = 12  
test_size = 0.30  

X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                    y_train,
                                                    test_size=test_size, 
                                                    random_state=state)

# tao the hien GradientBoostingClasssifier voi cac tham so 
# learning_rate cang thap cang chinh xac. Tuy nhien cung tang thoi gian tinh toan va truy van
# max_depth theo tai lieu mac dinh la 3. Gioi han do sau cua node 
# min_samples_leaf la so mau toi thieu trong moi node hoac la 
# min_samples_split la so mau toi thieu trong moi node duoc xem xet de phan tach
# random_state tao ra cac so ngau nhien 
gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=10)
gb.fit(X_train, y_train)
# du doan cho X_val
predictions = gb.predict(X_val)
print(confusion_matrix(y_val, predictions, labels=[1,0]))
print("Classification Report")
print(classification_report(y_val, predictions)) 

# tao mot mang learning rate voi cac so nhu ben duoi de kiem tra thong so nao la tot nhat
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=10)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

baseline = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, random_state=10)
baseline.fit(X_train,y_train)
# tao label la cho bieu do
X_label = [x for x in full_data.columns]
# tao bieu do bar co title la Importance of Features duoc sap xep giam dan dua theo cac gia tri quan trong 
feat_imp = pd.Series(baseline.feature_importances_, X_label).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
plt.show()