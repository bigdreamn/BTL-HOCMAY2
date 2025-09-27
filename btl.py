# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Khám phá và Tải Dữ liệu
#------------------------------------------------------------------------------------------------------
print("Bước 1: Khám phá và Tải Dữ liệu")
file_path = 'arxiv-metadata-oai-snapshot.json'
data_list = []
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # Giới hạn 50000 dòng để xử lý nhanh hơn, tránh quá tải bộ nhớ
        for i, line in enumerate(f):
           # if i >= 50000:
           #     break
            data_list.append(json.loads(line))
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Tạo DataFrame từ dữ liệu đã đọc
df = pd.DataFrame(data_list)
print(f"Số lượng mẫu được tải: {len(df)}")
print(f"Cấu trúc dữ liệu:\n{df.info()}")

# 2. Tiền xử lý văn bản
#------------------------------------------------------------------------------------------------------
print("\nBước 2: Tiền xử lý văn bản")
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Chuyển về chữ thường và loại bỏ các ký tự đặc biệt
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Loại bỏ các khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_abstract'] = df['abstract'].apply(preprocess_text)

# Tạo nhãn từ cột 'categories' và lọc các nhãn không phổ biến
df['label'] = df['categories'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'unknown')
label_counts = df['label'].value_counts()
# Giữ lại các nhãn có tần suất xuất hiện trên 100 lần
top_labels = label_counts[label_counts > 100].index
df_filtered = df[df['label'].isin(top_labels)].copy()
print(f"\nSố lượng mẫu sau khi lọc: {len(df_filtered)}")
print("Phân phối nhãn:\n", df_filtered['label'].value_counts())

# 3. Vector hóa văn bản với TF-IDF
#------------------------------------------------------------------------------------------------------
print("\nBước 3: Vector hóa văn bản với TF-IDF")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df_filtered['clean_abstract'])
y = df_filtered['label']
print(f"Kích thước ma trận TF-IDF: {X.shape}")

# 4. Huấn luyện mô hình Naive Bayes và Đánh giá
#------------------------------------------------------------------------------------------------------
print("\nBước 4: Huấn luyện mô hình Naive Bayes")
# Chia tập dữ liệu thành tập huấn luyện và kiểm tra với stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Đánh giá và Phân tích kết quả
#------------------------------------------------------------------------------------------------------
print("\nBước 5: Đánh giá và Phân tích kết quả")
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.4f}")
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred, zero_division=0))

# Trực quan hóa phân phối nhãn
plt.figure(figsize=(12, 6))
sns.countplot(y='label', data=df_filtered, order=df_filtered['label'].value_counts().index)
plt.title('Phân phối các nhãn (Categories)')
plt.xlabel('Số lượng')
plt.ylabel('Nhãn')
plt.tight_layout()
plt.show()

# Trực quan hóa ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred, labels=top_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_labels, yticklabels=top_labels)
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# 6. Tạo file solution.csv
#------------------------------------------------------------------------------------------------------
print("\nBước 6: Dự đoán và Tạo file solution.csv")
# Tải lại dữ liệu gốc để giữ thứ tự ID và các mẫu ban đầu
df_submission = df.copy()

# Áp dụng tiền xử lý và lọc tương tự như trên
df_submission['clean_abstract'] = df_submission['abstract'].apply(preprocess_text)
df_submission['label'] = df_submission['categories'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'unknown')

# Lọc các mẫu chỉ thuộc các nhãn đã được huấn luyện
df_submission = df_submission[df_submission['label'].isin(top_labels)].copy()

# Vector hóa toàn bộ dữ liệu đã lọc để dự đoán (dùng transform, không dùng fit_transform)
X_submission = tfidf_vectorizer.transform(df_submission['clean_abstract'])
y_submission = model.predict(X_submission)

# Tạo DataFrame kết quả
submission_df = pd.DataFrame({
    'id': df_submission['id'],
    'label': y_submission
})

# Lưu DataFrame ra file CSV
submission_df.to_csv('solution.csv', index=False)

print("Đã tạo file 'solution.csv' thành công.")
print("Vài dòng đầu tiên của file solution.csv:\n", submission_df.head())