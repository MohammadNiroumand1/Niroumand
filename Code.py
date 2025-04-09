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

def get_option_name(i):
    return f"گزینه {i+1}" if i > 0 else "گزینه مرجع"

# دریافت ورودی کاربر برای چند گزینه
options = []
while True:
    print(f"\nلطفاً اطلاعات {get_option_name(len(options))} را وارد کنید (یا 'end' برای پایان):")
    energy_quality = input("کیفیت معیار مصرف انرژی (EcNz, Ec++, Ec+, Ec): ")
    if energy_quality.lower() == 'end':
        break
    
    arch_perf = int(input("کیفیت الزامات عملکردی معماری (4, 3, 2, 1): "))
    maintenance_cost = int(input("کیفیت هزینه‌های نگهداری (4, 3, 2, 1): "))
    aesthetics = int(input("کیفیت زیبایی‌شناسی (4, 3, 2, 1): "))
    
    scores = np.array([
        score_energy(energy_quality),
        score_other(arch_perf),
        score_other(maintenance_cost),
        score_other(aesthetics)
    ])
    options.append(scores)

# اگر فقط یک گزینه وارد شده باشد، گزینه مرجع اضافه می‌کنیم
auto_generated = False
if len(options) == 1:
    print("\n⚠ فقط یک گزینه وارد شده است. گزینه مرجع با بالاترین امتیازات اضافه می‌شود.")
    reference_option = np.array([10, 10, 10, 10])  # بالاترین امتیازات ممکن
    options.append(reference_option)
    auto_generated = True

# 1. مدل وزن‌دهی و محاسبه نزدیکی نسبی
def build_weight_closeness_model():
    model = Sequential([
        Dense(64, input_shape=(4,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(5, activation='linear')  # 4 وزن + 1 نزدیکی نسبی
    ])
    # تابع Loss سفارشی
    def custom_loss(y_true, y_pred):
        weights_loss = tf.reduce_mean(tf.square(y_true[:, :4] - y_pred[:, :4]))
        closeness_loss = tf.reduce_mean(tf.square(y_true[:, 4] - y_pred[:, 4]) * y_true[:, 4])  # وزن بیشتر به نزدیکی نسبی بالا
        return weights_loss + closeness_loss
    model.compile(optimizer='adam', loss=custom_loss)
    return model

WEIGHT_CLOSENESS_MODEL_PATH = "weight_closeness_model.h5"
if not os.path.exists(WEIGHT_CLOSENESS_MODEL_PATH):
    weight_closeness_model = build_weight_closeness_model()
    X_train = np.random.uniform(2.5, 10, (5000, 4))
    y_train = np.zeros((5000, 5))
    for i in range(5000):
        normalized_scores = X_train[i] / np.sqrt(np.sum(X_train[i]**2))
        ideal_best = np.max(normalized_scores)
        ideal_worst = np.min(normalized_scores)
        S_best = np.linalg.norm(normalized_scores - ideal_best, ord=2)
        S_worst = np.linalg.norm(normalized_scores - ideal_worst, ord=2)
        closeness = 1 - (S_best / (S_best + S_worst)) * (1 + S_worst)
        y_train[i] = np.concatenate((X_train[i] / np.sum(X_train[i]), [closeness]))
    weight_closeness_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    weight_closeness_model.save(WEIGHT_CLOSENESS_MODEL_PATH)
else:
    weight_closeness_model = load_model(WEIGHT_CLOSENESS_MODEL_PATH)

# محاسبات برای هر گزینه
results = []
for i, scores in enumerate(options):
    weights_closeness = weight_closeness_model.predict(scores.reshape(1, -1), verbose=0)[0]
    weights = weights_closeness[:4] / np.sum(weights_closeness[:4])
    closeness = weights_closeness[4]
    
    normalized_scores = scores / np.sqrt(np.sum(scores**2))
    weighted_matrix = normalized_scores * weights
    
    if len(options) == 2 and i == 1 and auto_generated:
        ideal_best = weighted_matrix
        ideal_worst = np.zeros_like(weighted_matrix)
    else:
        ideal_best = np.max([opt / np.sqrt(np.sum(opt**2)) for opt in options], axis=0) * weights
        ideal_worst = np.min([opt / np.sqrt(np.sum(opt**2)) for opt in options], axis=0) * weights
    
    S_best = np.linalg.norm(weighted_matrix - ideal_best, ord=2)
    S_worst = np.linalg.norm(weighted_matrix - ideal_worst, ord=2)
    
    # اصلاح فرمول نزدیکی نسبی
    if auto_generated and i == 1:
        closeness = 1 - (S_best / (S_best + S_worst)) * (1 + S_worst) * (1 + (np.sum(scores) / 40))  # وزن بیشتر به امتیازات بالا
    
    results.append({
        'name': get_option_name(i),
        'scores': scores,
        'normalized': normalized_scores,
        'weights': weights,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    })

# محاسبه رتبه‌ها
closeness_ranks = np.argsort(-np.array([res['closeness'] for res in results])) + 1
S_best_ranks = np.argsort(np.array([res['S_best'] for res in results])) + 1
S_worst_ranks = np.argsort(-np.array([res['S_worst'] for res in results])) + 1

# نمایش نتایج
print("\n" + "="*50)
print("  نتایج مقایسه چندگزینه‌ای TOPSIS  ")
print("="*50)

for i, res in enumerate(results):
    print(f"\n {res['name']}:")
    print(f"- امتیازات خام: {res['scores']}")
    print(f"- مقادیر نرمال‌شده: {res['normalized'].round(4)}")
    print(f"- وزن‌ها: {res['weights'].round(4)}")
    print(f"- مقادیر وزنی: {res['weighted_matrix'].round(4)}")
    print(f"- فاصله از ایده‌آل مثبت: {res['S_best']:.4f} (رتبه: {S_best_ranks[i]})")
    print(f"- فاصله از ایده‌آل منفی: {res['S_worst']:.4f} (رتبه: {S_worst_ranks[i]})")
    print(f"- نزدیکی نسبی: {res['closeness']:.4f} (رتبه: {closeness_ranks[i]})")
    
    if res['closeness'] >= 0.9:
        rating = "⭐⭐⭐⭐⭐ گزینه ممتاز"
    elif res['closeness'] >= 0.75:
        rating = "⭐⭐⭐⭐ گزینه عالی"
    elif res['closeness'] >= 0.6:
        rating = "⭐⭐⭐ گزینه خوب"
    elif res['closeness'] >= 0.4:
        rating = "⭐⭐ گزینه متوسط"
    else:
        rating = "⭐ گزینه ضعیف"
    print(f"- رتبه‌بندی: {rating}")

print("\n" + "="*50)
if len(results) > 1:
    best_option = max(results, key=lambda x: x['closeness'])
    print(f"\n بهترین گزینه: {best_option['name']} با نزدیکی نسبی {best_option['closeness']:.4f}")
print("="*50)
