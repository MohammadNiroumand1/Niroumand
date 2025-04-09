import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os

# تنظیمات TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

# توابع امتیازدهی
def score_energy(quality):
    qualities = {'EcNz': 10, 'Ec++': 7.5, 'Ec+': 5, 'Ec': 2.5}
    return qualities.get(quality, 0)

def score_other(quality):
    qualities = {4: 10, 3: 7.5, 2: 5, 1: 2.5}
    return qualities.get(quality, 0)

# دریافت ورودی کاربر
print("\nلطفاً اطلاعات گزینه را وارد کنید:")
energy_quality = input("کیفیت معیار مصرف انرژی (EcNz, Ec++, Ec+, Ec): ")
arch_perf = int(input("کیفیت الزامات عملکردی معماری (4, 3, 2, 1): "))
maintenance_cost = int(input("کیفیت هزینه‌های نگهداری (4, 3, 2, 1): "))
aesthetics = int(input("کیفیت زیبایی‌شناسی (4, 3, 2, 1): "))

# ماتریس تصمیم
scores = np.array([
    score_energy(energy_quality),
    score_other(arch_perf),
    score_other(maintenance_cost),
    score_other(aesthetics)
])

# 1. مدل وزن‌دهی ساده
def build_weight_model():
    model = Sequential([
        Dense(32, input_shape=(4,), activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

WEIGHT_MODEL_PATH = "weights_model.h5"
if not os.path.exists(WEIGHT_MODEL_PATH):
    weight_model = build_weight_model()
    X_train = np.random.uniform(2.5, 10, (5000, 4))
    y_train = X_train / np.sum(X_train, axis=1, keepdims=True)
    weight_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    weight_model.save(WEIGHT_MODEL_PATH)
else:
    weight_model = load_model(WEIGHT_MODEL_PATH)

weights = weight_model.predict(scores.reshape(1, -1), verbose=0)[0]

# 2. محاسبات TOPSIS ساده شده
# نرمال‌سازی
normalized_scores = scores / np.sqrt(np.sum(scores**2))

# ماتریس وزنی
weighted_matrix = normalized_scores * weights

# راه‌حل‌های ایده‌آل
ideal_best = np.ones(4) * weights  # همه معیارها حداکثر (1 پس از نرمال‌سازی)
ideal_worst = np.zeros(4) * weights  # همه معیارها حداقل (0 پس از نرمال‌سازی)

# محاسبه فواصل
S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2))
S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2))

# 3. مدل شبکه عصبی ساده برای محاسبه نزدیکی نسبی
def build_closeness_model():
    model = Sequential([
        Dense(16, input_shape=(6,), activation='relu'),  # 4 وزن + 2 فاصله
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

CLOSENESS_MODEL_PATH = "closeness_model.h5"
if not os.path.exists(CLOSENESS_MODEL_PATH):
    closeness_model = build_closeness_model()
    
    # تولید داده‌های آموزشی ساده
    X_train = []
    y_train = []
    
    for _ in range(5000):
        # تولید امتیازهای تصادفی
        current_scores = np.random.uniform(2.5, 10, 4)
        
        # محاسبه وزن‌ها
        current_weights = current_scores / np.sum(current_scores)
        
        # شبیه‌سازی فواصل
        sum_scores = np.sum(current_scores)
        S_best_sim = max(0.01, 1 - (sum_scores/40))  # رابطه معکوس
        S_worst_sim = min(0.99, sum_scores/40)  # رابطه مستقیم
        
        # هدف: نزدیک به 1 برای امتیازهای بالا
        target = min(0.99, sum_scores/40 + 0.5)  # بین 0.5 تا 0.99
        
        X_train.append(np.concatenate([current_weights, [S_best_sim, S_worst_sim]]))
        y_train.append(target)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    closeness_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    closeness_model.save(CLOSENESS_MODEL_PATH)
else:
    closeness_model = load_model(CLOSENESS_MODEL_PATH)

# محاسبه نزدیکی نسبی با تقویت پیشرفته
model_input = np.concatenate([weights, [S_best, S_worst]])
closeness = closeness_model.predict(model_input.reshape(1, -1), verbose=0)[0][0]

# تقویت پیشرفته نتیجه
sum_scores = np.sum(scores)
max_score = np.max(scores)
min_score = np.min(scores)

# 1. تقویت مبتنی بر مجموع امتیازها
closeness *= (1 + (sum_scores - 10)/50)  # تقویت بیشتر برای امتیازهای بالا

# 2. تقویت مبتنی بر حداکثر امتیاز
closeness *= (1 + (max_score - 2.5)/30)

# 3. تقویت مبتنی بر حداقل امتیاز (کاهش اثر امتیازهای پایین)
closeness *= (1 + (min_score - 2.5)/40)

# 4. اعمال تابع توانی برای نزدیک کردن به 1
closeness = 1 - (1 - closeness)**0.7

# 5. محدود کردن به بازه 0.01 تا 0.99
closeness = max(0.01, min(0.99, closeness))

# 6. تقویت نهایی بر اساس توزیع وزن‌ها
weight_factor = np.max(weights) / np.mean(weights)
closeness = min(0.99, closeness * (1 + weight_factor/10))

# نمایش نتایج
print("\n" + "="*50)
print("  محاسبه عصبی  نزدیکی نسبی")
print("="*50)

print("\n🔹 امتیازهای خام ورودی:")
print(f"- مصرف انرژی: {scores[0]}")
print(f"- عملکرد معماری: {scores[1]}")
print(f"- هزینه نگهداری: {scores[2]}")
print(f"- زیبایی‌شناسی: {scores[3]}")

print("\n🔹 مقادیر نرمال‌سازی شده (با روش TOPSIS):")
print(f"- مصرف انرژی: {normalized_scores[0]:.4f}")
print(f"- عملکرد معماری: {normalized_scores[1]:.4f}")
print(f"- هزینه نگهداری: {normalized_scores[2]:.4f}")
print(f"- زیبایی‌شناسی: {normalized_scores[3]:.4f}")

print("\n🔹 وزن‌های محاسبه شده:")
print(f"- مصرف انرژی: {weights[0]:.4f}")
print(f"- عملکرد معماری: {weights[1]:.4f}")
print(f"- هزینه نگهداری: {weights[2]:.4f}")
print(f"- زیبایی‌شناسی: {weights[3]:.4f}")

print("\n🔹 مقادیر وزنی (نرمال‌شده × وزن):")
print(f"- مصرف انرژی: {weighted_matrix[0]:.4f}")
print(f"- عملکرد معماری: {weighted_matrix[1]:.4f}")
print(f"- هزینه نگهداری: {weighted_matrix[2]:.4f}")
print(f"- زیبایی‌شناسی: {weighted_matrix[3]:.4f}")

print("\n🔹 فواصل محاسبه شده:")
print(f"- از راه‌حل ایده‌آل مثبت: {S_best:.4f}")
print(f"- از راه‌حل ایده‌آل منفی: {S_worst:.4f}")

print("\n🔹 نتیجه نهایی:")
print(f"- نزدیکی نسبی : {closeness:.4f}")

print("\n" + "="*50)
if closeness >= 0.9:
    print("⭐⭐⭐⭐⭐ گزینه ممتاز (نزدیکی ≥ 0.9)")
elif closeness >= 0.75:
    print("⭐⭐⭐⭐ گزینه عالی (0.75 ≤ نزدیکی < 0.9)")
elif closeness >= 0.6:
    print("⭐⭐⭐ گزینه خوب (0.6 ≤ نزدیکی < 0.75)")
elif closeness >= 0.4:
    print("⭐⭐ گزینه متوسط (0.4 ≤ نزدیکی < 0.6)")
else:
    print("⭐ گزینه ضعیف (نزدیکی < 0.4)")
print("="*50)
