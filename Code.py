import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import os

# ثابت کردن seed برای نتایج یکسان
np.random.seed(42)
tf.random.set_seed(42)

# تعریف توابع امتیازدهی
def score_energy(quality):
    if quality == 'EcNz':
        return 10
    elif quality == 'Ec++':
        return 7.5
    elif quality == 'Ec+':
        return 5
    elif quality == 'Ec':
        return 2.5
    else:
        return 0

def score_other(quality):
    if quality == 4:
        return 10
    elif quality == 3:
        return 7.5
    elif quality == 2:
        return 5
    elif quality == 1:
        return 2.5
    else:
        return 0

# دریافت ورودی‌ها از کاربر برای یک گزینه
print("لطفاً اطلاعات گزینه را وارد کنید:")
energy_quality = input("کیفیت معیار مصرف انرژی را وارد کنید (EcNz, Ec++, Ec+, Ec): ")
arch_perf = int(input("کیفیت الزامات عملکردی معماری را وارد کنید (4, 3, 2, 1): "))
maintenance_cost = int(input("کیفیت هزینه‌های نگهداری را وارد کنید (4, 3, 2, 1): "))
aesthetics = int(input("کیفیت زیبایی‌شناسی را وارد کنید (4, 3, 2, 1): "))

# امتیازدهی به معیارها
energy_score = score_energy(energy_quality)
arch_perf_score = score_other(arch_perf)
maintenance_cost_score = score_other(maintenance_cost)
aesthetics_score = score_other(aesthetics)

# ایجاد ماتریس تصمیم (تنها یک گزینه)
decision_matrix = np.array([energy_score, arch_perf_score, maintenance_cost_score, aesthetics_score])

# نگاشت امتیاز به وزن
raw_weight_map = {
    10: 4.0,
    7.5: 3.0,
    5: 2.0,
    2.5: 1.0,
    0: 0.0,
    2.25: 0.5
}

# محاسبه وزن‌های خام
raw_weights = np.array([raw_weight_map[score] for score in decision_matrix])

# نرمال‌سازی وزن‌ها به جمع 1 (فقط برای نمایش)
normalized_weights = raw_weights / np.sum(raw_weights) if np.sum(raw_weights) > 0 else np.zeros_like(raw_weights)

# نرمال‌سازی ماتریس تصمیم
matrix_norm = np.linalg.norm(decision_matrix)
normalized_matrix = decision_matrix / matrix_norm if matrix_norm > 0 else np.zeros_like(decision_matrix)

# نمایش داده‌های نرمال‌سازی شده
print("\nماتریس نرمال‌سازی شده:")
print(normalized_matrix)

# نمایش وزن‌های محاسبه شده برای معیارها
print("\nوزن‌های نرمال‌سازی شده برای معیارها (جمع=1):")
print(f"مصرف انرژی: {normalized_weights[0]:.4f}")
print(f"عملکرد معماری: {normalized_weights[1]:.4f}")
print(f"هزینه نگهداری: {normalized_weights[2]:.4f}")
print(f"زیبایی‌شناسی: {normalized_weights[3]:.4f}")

# اعمال وزن‌ها به ماتریس نرمال‌سازی شده
weighted_matrix = normalized_matrix * raw_weights

# محاسبه فاصله از راه‌حل‌های ایده‌آل با روش TOPSIS
print("\nمحاسبه فاصله از راه‌حل‌های ایده‌آل (TOPSIS):")

# راه‌حل ایده‌آل مثبت (بهترین مقادیر ممکن)
ideal_positive = np.array([10, 10, 10, 10])
# راه‌حل ایده‌آل منفی (بدترین مقادیر ممکن)
ideal_negative = np.array([0, 0, 0, 0])

# نرمال‌سازی راه‌حل‌های ایده‌آل
ideal_pos_norm = ideal_positive / np.linalg.norm(ideal_positive)
ideal_neg_norm = ideal_negative / np.linalg.norm(ideal_negative) if np.linalg.norm(ideal_negative) > 0 else np.zeros_like(ideal_negative)

# محاسبه فاصله وزنی از راه‌حل‌های ایده‌آل
weighted_ideal_pos = ideal_pos_norm * raw_weights
weighted_ideal_neg = ideal_neg_norm * raw_weights

# محاسبه فاصله اقلیدسی با محدودیت بازه
distance_positive = np.clip(np.linalg.norm(weighted_matrix - weighted_ideal_pos), 0, None)
distance_negative = np.clip(np.linalg.norm(weighted_matrix - weighted_ideal_neg), 0, None)

# محاسبه حداکثر فاصله ممکن با محدودیت
max_distance = np.clip(np.linalg.norm(weighted_ideal_pos - weighted_ideal_neg), 1e-10, None)

# نرمال‌سازی فاصله‌ها با تضمین بازه 0-1
normalized_distance_positive = np.clip(distance_positive / max_distance, 0.0, 1.0)
normalized_distance_negative = np.clip(distance_negative / max_distance, 0.0, 1.0)

# نمایش نتایج با اطمینان از محدوده 0-1
print(f"فاصله نرمال‌سازی شده از راه‌حل ایده‌آل مثبت: {normalized_distance_positive:.4f} (هدف: نزدیک به 0)")
print(f"فاصله نرمال‌سازی شده از راه‌حل ایده‌آل منفی: {normalized_distance_negative:.4f} (هدف: نزدیک به 1)")

# مسیر ذخیره مدل
MODEL_PATH = "trained_model.h5"

# بررسی وجود مدل از قبل آموزش دیده
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    # تعریف مدل جدید اگر وجود نداشت
    model = Sequential([
        Dense(64, input_shape=(8,), activation='relu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # تولید داده‌های آموزشی ثابت
    train_scores = [10, 7.5, 5, 2.5, 2.25, 0]
    X_scores = np.array(np.meshgrid(train_scores, train_scores, train_scores, train_scores)).T.reshape(-1, 4)
    X_weights = np.array([[raw_weight_map[score] for score in row] for row in X_scores])
    X_train = np.concatenate([X_scores, X_weights], axis=1)
    weighted_scores = X_scores * X_weights
    sum_weighted = np.sum(weighted_scores, axis=1)
    sum_weights = np.sum(X_weights, axis=1)
    y_train = (sum_weighted / (sum_weights + 1e-10)) / 10
    
    # آموزش مدل
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=0)
    
    # ذخیره مدل آموزش دیده
    model.save(MODEL_PATH)

# پیش‌بینی با نتایج ثابت
input_data = np.concatenate([decision_matrix.reshape(1,-1), raw_weights.reshape(1,-1)], axis=1).astype(np.float32)
predicted_closeness = model.predict(input_data, verbose=0)[0][0]

# نمایش نتایج
print("\nنزدیکی نسبی پیش‌بینی شده به راه‌حل ایده‌آل:")
print(f"نتیجه: {predicted_closeness:.4f}")
