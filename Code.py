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

# دریافت ورودی کاربر
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

# اضافه کردن گزینه مرجع اگر نیاز باشد
auto_generated = False
if len(options) == 1:
    print("\n⚠ فقط یک گزینه وارد شده است. گزینه مرجع با بالاترین امتیازات اضافه می‌شود.")
    reference_option = np.array([10, 10, 10, 10])
    options.append(reference_option)
    auto_generated = True

# مدل شبکه عصبی برای وزن‌دهی
def build_weight_model():
    model = Sequential([
        Dense(64, input_shape=(4,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')  # وزن‌های نرمال‌شده
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

WEIGHT_MODEL_PATH = "weight_model.h5"
if not os.path.exists(WEIGHT_MODEL_PATH):
    weight_model = build_weight_model()
    # آموزش مدل با وزن‌های متوازن
    X_train = np.random.uniform(2.5, 10, (5000, 4))
    y_train = np.ones((5000, 4)) * 0.25  # هدف: وزن‌های برابر
    weight_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    weight_model.save(WEIGHT_MODEL_PATH)
else:
    weight_model = load_model(WEIGHT_MODEL_PATH)

# محاسبات TOPSIS
results = []
all_scores = np.array(options)

# نرمال‌سازی وزنی
normalized_matrix = all_scores / np.sqrt(np.sum(all_scores**2, axis=1, keepdims=True))

# محاسبه وزن‌ها با مدل عصبی
weights = weight_model.predict(normalized_matrix, verbose=0)[0]
weights = weights / np.sum(weights)  # اطمینان از جمع برابر با 1

# ماتریس وزنی
weighted_matrix = normalized_matrix * weights

# محاسبه ایده‌آل‌ها
ideal_best = np.max(weighted_matrix, axis=0)
ideal_worst = np.min(weighted_matrix, axis=0)

for i, scores in enumerate(options):
    # محاسبه فواصل
    S_best = np.linalg.norm(weighted_matrix[i] - ideal_best, ord=2)
    S_worst = np.linalg.norm(weighted_matrix[i] - ideal_worst, ord=2)
    
    # محاسبه نزدیکی نسبی به روش سنتی
    closeness = S_worst / (S_best + S_worst)
    
    results.append({
        'name': get_option_name(i),
        'scores': scores,
        'normalized': normalized_matrix[i],
        'weights': weights,
        'weighted_matrix': weighted_matrix[i],
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
print("  نتایج مقایسه چندگزینه‌ای TOPSIS (با شبکه عصبی)  ")
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
    
    # رتبه‌بندی
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

# انتخاب بهترین گزینه
if len(results) > 1:
    best_option = max(results, key=lambda x: x['closeness'])
    print("\n" + "="*50)
    print(f" بهترین گزینه: {best_option['name']} با نزدیکی نسبی {best_option['closeness']:.4f}")
    print("="*50)
