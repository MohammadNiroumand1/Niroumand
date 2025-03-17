Python 3.5.1 (v3.5.1:37a07cee5969, Dec  6 2015, 01:38:48) [MSC v.1900 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

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
weight_map = {
    10: 0.25,
    7.5: 0.1875,
    5: 0.125,
    2.5: 0.0625,
    0: 0,
    2.25: 0.03125
}

# محاسبه وزن‌ها بر اساس نگاشت
weights = np.array([weight_map[score] for score in decision_matrix])

# نرمال‌سازی ماتریس تصمیم
normalized_matrix = decision_matrix / np.linalg.norm(decision_matrix)

# نمایش داده‌های نرمال‌سازی شده
print("\nماتریس نرمال‌سازی شده:")
print(normalized_matrix)

# اعمال وزن‌ها به ماتریس نرمال‌سازی شده
weighted_matrix = normalized_matrix * weights

# نمایش وزن‌های محاسبه شده برای معیارها
print("\nوزن‌های محاسبه شده برای معیارها:")
print(f"مصرف انرژی: {weights[0]:.4f}")
print(f"عملکرد معماری: {weights[1]:.4f}")
print(f"هزینه نگهداری: {weights[2]:.4f}")
print(f"زیبایی‌شناسی: {weights[3]:.4f}")

# تعریف راه‌حل‌های ایده‌آل
ideal_positive = np.array([0.0] * 4)  # راه‌حل ایده‌آل مثبت (تمام عناصر 1)
ideal_negative = np.array([1.0] * 4)  # راه‌حل ایده‌آل منفی (تمام عناصر 0)

# محاسبه فاصله از راه‌حل‌های ایده‌آل
distance_positive = np.linalg.norm(weighted_matrix - ideal_positive)
distance_negative = np.linalg.norm(weighted_matrix - ideal_negative)

# نرمال‌سازی فاصله برای بازه 0 تا 1
max_distance = np.linalg.norm(ideal_positive - ideal_negative)  # حداکثر فاصله ممکن
distance_positive_normalized = distance_positive / max_distance  # فاصله از راه‌حل ایده‌آل مثبت
distance_negative_normalized = 1 - (distance_negative / max_distance)  # فاصله از راه‌حل ایده‌آل منفی

# نمایش نتایج فاصله
print("\nفاصله از راه‌حل ایده‌آل مثبت (0 تا 1):")
print(f"فاصله: {distance_positive_normalized:.4f}")

print("\nفاصله از راه‌حل ایده‌آل منفی (0 تا 1):")
print(f"فاصله: {distance_negative_normalized:.4f}")

# تعریف مدل شبکه عصبی برای پیش‌بینی نزدیکی نسبی
model = Sequential([
    Dense(32, input_shape=(4,), activation='relu', kernel_regularizer=l2(0.01)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')  # استفاده از تابع فعال‌سازی sigmoid برای خروجی 0 تا 1
])

# کامپایل مدل
model.compile(optimizer='adam', loss='mse')

# تولید داده‌های آموزشی بر اساس امتیازات ممکن
train_scores = [10, 7.5, 5, 2.25]
num_samples = len(train_scores) ** 4
X_train = np.array(np.meshgrid(train_scores, train_scores, train_scores, train_scores)).T.reshape(-1, 4)

# محاسبه وزن‌ها بر اساس نگاشت
weights_train = np.array([[weight_map[score] for score in row] for row in X_train])

# محاسبه مجموع وزن‌ها برای هر نمونه آموزشی
total_weights_train = np.sum(weights_train, axis=1)

# نرمال‌سازی مجموع وزن‌ها برای استفاده در تابع سیگموئید
normalized_total_weights_train = total_weights_train / np.max(total_weights_train)

# محاسبه نزدیکی نسبی با استفاده از تابع سیگموئید
y_train = 1 / (1 + np.exp(-10 * normalized_total_weights_train + 5))

# آموزش مدل
model.fit(X_train, y_train, epochs=200, verbose=0)

# پیش‌بینی نزدیکی نسبی برای گزینه وارد شده
total_weight = np.sum(weights)
normalized_total_weight = total_weight / np.max(np.sum(weights_train, axis=1))
predicted_closeness = 1 / (1 + np.exp(-10 * normalized_total_weight + 5))

# نمایش نتایج نزدیکی نسبی
print("\nنزدیکی نسبی پیش‌بینی شده به راه‌حل ایده‌آل:")
print(f"نتیجه: {predicted_closeness:.4f}")
