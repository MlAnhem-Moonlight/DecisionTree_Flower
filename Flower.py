import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Hàm đọc dữ liệu và xử lý
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Xử lý dữ liệu train
    label_encoder = LabelEncoder()
    train_data['leaf_shape'] = label_encoder.fit_transform(train_data['leaf_shape'])
    train_data['flower'] = label_encoder.fit_transform(train_data['flower'])

    # Test data không chứa nhãn "flower", chỉ xử lý các cột đặc trưng
    test_data['leaf_shape'] = LabelEncoder().fit_transform(test_data['leaf_shape'])

    # Chọn đặc trưng và nhãn
    X_train = train_data[['petal_count', 'petal_length', 'petal_width', 'stem_thorns', 'leaf_shape', 'scent_intensity', 'flower_size']]
    y_train = train_data['flower']

    # Chỉ lấy đặc trưng từ file test
    X_test = test_data[['petal_count', 'petal_length', 'petal_width', 'stem_thorns', 'leaf_shape', 'scent_intensity', 'flower_size']]

    return X_train, y_train, X_test, label_encoder


# Hàm huấn luyện và dự đoán
def train_and_predict(train_file, test_file, max_depth, min_samples_split):
    # Bước 1: Đọc dữ liệu
    X_train, y_train, X_test, label_encoder = load_data(train_file, test_file)

    # Bước 2: Huấn luyện cây quyết định
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)

    # Bước 3: Dự đoán nhãn cho dữ liệu test
    y_pred = clf.predict(X_test)

    # Chuyển từ số sang tên nhãn hoa
    flower_names = label_encoder.inverse_transform(y_pred)

    # In ra dự đoán cho từng mẫu trong file test
    print("Dự đoán loại hoa cho file test:")
    for i, flower in enumerate(flower_names):
        print(f"Mẫu {i + 1}: Dự đoán là hoa {flower}")

    # Bước 4: Visualize cây quyết định (tuỳ chọn)
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=X_train.columns, class_names=label_encoder.classes_, filled=True)
    plt.show()

    # Xuất cây sang file .dot (nếu muốn dùng Graphviz)
    export_graphviz(
        clf,
        out_file="tree.dot",
        feature_names=X_train.columns,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True
    )
    print("Tree exported to tree.dot. You can visualize it using Graphviz.")


# Chạy chương trình
if __name__ == "__main__":
    # Thay đường dẫn file
    train_file = "flower_train.csv"  # File training (có nhãn)
    test_file = "flower_test.csv"    # File testing (không có nhãn)

    # Nhập các tham số từ người dùng
    max_depth = int(input("Nhập độ sâu tối đa cho cây quyết định (max_depth): "))
    min_samples_split = int(input("Nhập số mẫu tối thiểu tại mỗi node (min_samples_split): "))

    # Gọi hàm huấn luyện và dự đoán
    train_and_predict(train_file, test_file, max_depth, min_samples_split)
