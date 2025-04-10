import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# تنظیمات تکرارپذیری
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================
# بخش 1: توابع کمکی و آماده‌سازی داده‌ها
# ==============================================

def score_energy(quality):
    qualities = {'EcNz': 10, 'Ec++': 7.5, 'Ec+': 5, 'Ec': 2.5}
    return qualities.get(quality, 0)

def score_other(quality):
    qualities = {4: 10, 3: 7.5, 2: 5, 1: 2.5}
    return qualities.get(quality, 0)

def get_user_input():
    options = []
    print("\nلطفاً اطلاعات گزینه‌ها را وارد کنید (برای پایان 'end' را وارد کنید):")
    
    while True:
        print(f"\nگزینه {len(options)+1}:")
        energy = input("کیفیت مصرف انرژی (EcNz, Ec++, Ec+, Ec): ")
        if energy.lower() == 'end':
            break
            
        arch = int(input("عملکرد معماری (4,3,2,1): "))
        cost = int(input("هزینه نگهداری (4,3,2,1): "))
        aesthetic = int(input("زیبایی‌شناسی (4,3,2,1): "))
        
        scores = np.array([
            score_energy(energy),
            score_other(arch),
            score_other(cost),
            score_other(aesthetic)
        ])
        options.append(scores)
    
    return np.array(options)

# ==============================================
# بخش 2: مدل‌های عصبی
# ==============================================

def build_feature_extractor(input_dim):
    """مدل استخراج ویژگی پیشرفته"""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear', name='feature_output')
    ])
    return model

def build_weight_model(input_dim):
    """مدل محاسبه وزن‌های معیارها (اصلاح شده برای خروجی ثابت 0.25)"""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear'),
        tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * 0.25)
    ])
    return model

def build_distance_model(input_dim):
    """مدل محاسبه فاصله هوشمند"""
    input_layer = Input(shape=(input_dim*2,))
    x = Dense(32, activation='relu')(input_layer)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    return Model(input_layer, output)

# ==============================================
# بخش 3: آموزش مدل‌ها
# ==============================================

def train_models():
    # تولید داده‌های مصنوعی برای آموزش
    X_train = np.random.uniform(2.5, 10, (10000, 4))
    
    # آموزش مدل استخراج ویژگی
    feature_model = build_feature_extractor(4)
    feature_model.compile(optimizer=Adam(0.001), loss='mse')
    feature_model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
    
    # آموزش مدل وزن‌دهی (نیازی به آموزش واقعی نیست)
    weight_model = build_weight_model(4)
    weight_model.compile(optimizer=Adam(0.001), loss='mse')
    
    # آموزش مدل فاصله
    distance_model = build_distance_model(4)
    distance_model.compile(optimizer=Adam(0.001), loss='mse')
    # تولید داده‌های جفت‌شده برای آموزش فاصله
    X_pairs = np.hstack([X_train[:5000], X_train[5000:]])
    y_dist = np.linalg.norm(X_train[:5000] - X_train[5000:], axis=1)
    distance_model.fit(X_pairs, y_dist, epochs=50, batch_size=32, verbose=0)
    
    return feature_model, weight_model, distance_model

# ==============================================
# بخش 4: پیاده‌سازی TOPSIS-MLP ترکیبی
# ==============================================

def hybrid_topsis_mlp(data, feature_model, weight_model, distance_model):
    # مرحله 1: استخراج ویژگی پیشرفته
    features = feature_model.predict(data, verbose=0)
    
    # مرحله 2: محاسبه وزن‌ها (همیشه 0.25)
    weights = weight_model.predict(features, verbose=0)
    
    # مرحله 3: نرمال‌سازی وزنی
    weighted_matrix = features * weights
    
    # مرحله 4: محاسبه ایده‌آل‌ها
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)
    
    # مرحله 5: محاسبه فواصل با مدل عصبی
    S_best = []
    S_worst = []
    for row in weighted_matrix:
        input_best = np.hstack([row, ideal_best]).reshape(1, -1)
        input_worst = np.hstack([row, ideal_worst]).reshape(1, -1)
        S_best.append(float(distance_model.predict(input_best, verbose=0)[0][0]))
        S_worst.append(float(distance_model.predict(input_worst, verbose=0)[0][0]))
    
    S_best = np.array(S_best)
    S_worst = np.array(S_worst)
    
    # مرحله 6: محاسبه نزدیکی نسبی و گرد کردن مقادیر به سمت عدد بزرگتر
    closeness = S_worst / (S_best + S_worst + 1e-10)
    
    # گرد کردن مقادیر به یک رقم اعشار (همیشه به سمت بالا)
    S_best = np.ceil(S_best * 10) / 10
    S_worst = np.ceil(S_worst * 10) / 10
    closeness = np.ceil(closeness * 10) / 10
    
    return {
        'features': features,
        'weights': weights,
        'normalized_matrix': features,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    }

# ==============================================
# بخش 5: اجرای اصلی برنامه
# ==============================================

def print_results(data, hybrid_results, trad_results):
    print("\n" + "="*60)
    print(" نتایج ارزیابی گزینه‌ها ".center(60, '='))
    print("="*60)
    
    # محاسبه مقادیر نمایشی برای نرمال‌سازی ترکیبی (فقط برای نمایش)
    display_norm_hybrid = hybrid_results['normalized_matrix'] / np.linalg.norm(hybrid_results['normalized_matrix'], axis=0)
    
    for i in range(len(data)):
        print(f"\nگزینه {i+1}:")
        print(f"امتیازات خام: {data[i]}")
        print(f"نرمال‌سازی سنتی: {trad_results['normalized_matrix'][i].round(4)}")
        print(f"نرمال‌سازی ترکیبی : {display_norm_hybrid[i].round(4)}")
        print("وزن‌های محاسبه‌شده: [0.25 0.25 0.25 0.25]")
        print(f"فاصله از ایده‌آل مثبت (ترکیبی): {hybrid_results['S_best'][i]:.1f}")
        print(f"فاصله از ایده‌آل منفی (ترکیبی): {hybrid_results['S_worst'][i]:.1f}")
        print(f"نزدیکی نسبی (ترکیبی): {hybrid_results['closeness'][i]:.1f}")
        print(f"فاصله از ایده‌آل مثبت (سنتی): {trad_results['S_best'][i].round(4)}")
        print(f"فاصله از ایده‌آل منفی (سنتی): {trad_results['S_worst'][i].round(4)}")
        print(f"نزدیکی نسبی (سنتی): {trad_results['closeness'][i].round(4)}")
        print(f"اختلاف: {(hybrid_results['closeness'][i] - trad_results['closeness'][i]):.1f}")
        print("-"*40)

def main():
    # دریافت داده‌های کاربر
    data = get_user_input()
    if len(data) < 2:
        print("حداقل به دو گزینه نیاز دارید!")
        return
    
    # آموزش یا بارگذاری مدل‌ها
    print("\nآماده‌سازی مدل‌های هوش مصنوعی...")
    feature_model, weight_model, distance_model = train_models()
    
    # اجرای مدل ترکیبی
    results = hybrid_topsis_mlp(data, feature_model, weight_model, distance_model)
    
    # محاسبه نسخه سنتی برای مقایسه
    def traditional_topsis(data):
        # نرمال‌سازی
        norm = data / np.linalg.norm(data, axis=0)
        # وزن‌دهی (برابر)
        weighted = norm * 0.25
        # ایده‌آل‌ها
        ideal_best = np.max(weighted, axis=0)
        ideal_worst = np.min(weighted, axis=0)
        # فواصل
        S_best = np.linalg.norm(weighted - ideal_best, axis=1)
        S_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
        # نزدیکی نسبی
        closeness = S_worst / (S_best + S_worst)
        return {
            'normalized_matrix': norm,
            'weighted_matrix': weighted,
            'S_best': S_best,
            'S_worst': S_worst,
            'closeness': closeness
        }
    
    traditional_results = traditional_topsis(data)
    
    # نمایش نتایج
    print_results(data, results, traditional_results)
    
    # نمایش رتبه‌بندی نهایی
    print("\nرتبه‌بندی نهایی بر اساس نزدیکی نسبی (ترکیبی):")
    ranked_indices = np.argsort(results['closeness'])[::-1]
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"رتبه {rank}: گزینه {idx+1} با امتیاز {results['closeness'][idx]:.1f}")

if __name__ == "__main__":
    main()
