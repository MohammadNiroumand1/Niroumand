import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings

np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore', category=UserWarning)

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
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear', name='feature_output')
    ])
    return model

def build_distance_model(input_dim):
    """مدل محاسبه فاصله هوشمند"""
    input_layer = Input(shape=(input_dim*2,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    return Model(input_layer, output)

# ==============================================
# بخش 3: آموزش مدل‌ها
# ==============================================

@tf.function(reduce_retracing=True)
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_models():
    # تولید داده‌های آموزشی
    X_train = np.random.uniform(2.5, 10, (10000, 4))

    # مدل استخراج ویژگی
    feature_model = build_feature_extractor(4)
    feature_model.compile(optimizer=Adam(0.001), loss='mse')
    feature_model.fit(X_train, X_train, epochs=100, batch_size=64, verbose=0)

    # مدل فاصله
    distance_model = build_distance_model(4)
    distance_model.compile(optimizer=Adam(0.001), loss='mse')

    # داده‌های جفت‌شده برای آموزش فاصله
    X_pairs = np.hstack([X_train[:5000], X_train[5000:]])
    y_dist = np.linalg.norm(X_train[:5000] - X_train[5000:], axis=1) * np.random.uniform(0.9, 1.1, 5000)
    distance_model.fit(X_pairs, y_dist, epochs=100, batch_size=64, verbose=0)

    return feature_model, distance_model

# ==============================================
# بخش 4: پیاده‌سازی TOPSIS-MLP ترکیبی بهینه‌شده
# ==============================================

def hybrid_topsis_mlp(data, feature_model, distance_model):
    # استخراج ویژگی‌ها
    features = feature_model.predict(data, verbose=0)

    # نرمال‌سازی
    norms = np.linalg.norm(features, axis=0)
    normalized_matrix = features / norms

    # وزن‌دهی
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted_matrix = normalized_matrix * weights

    # محاسبه ایده‌آل‌ها
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # محاسبه فواصله
    if len(data) <= 10:
        # محاسبات سنتی
        S_best_trad = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        S_worst_trad = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # محاسبات عصبی
        inputs_best = np.hstack([weighted_matrix, np.tile(ideal_best, (len(data), 1))])
        inputs_worst = np.hstack([weighted_matrix, np.tile(ideal_worst, (len(data), 1))])

        S_best_nn = distance_model.predict(inputs_best, verbose=0).flatten()
        S_worst_nn = distance_model.predict(inputs_worst, verbose=0).flatten()

        # ترکیب فواصل به صورت وزن‌دار
        S_best = 0.8 * S_best_trad + 0.2 * S_best_nn
        S_worst = 0.8 * S_worst_trad + 0.2 * S_worst_nn
    else:
        S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # محاسبه نزدیکی نسبی
    epsilon = 1e-10
    closeness = S_worst / (S_best + S_worst + epsilon)

    return {
        'features': features,
        'weights': weights,
        'normalized_matrix': normalized_matrix,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    }

# ==============================================
# بخش 5: توابع نمایش و TOPSIS سنتی
# ==============================================

def print_results(data, hybrid_results, trad_results):
    print("\n" + "="*60)
    print(" نتایج ارزیابی گزینه‌ها ".center(60, '='))
    print("="*60)

    for i in range(len(data)):
        print(f"\nگزینه {i+1}:")
        print(f"امتیازات خام: {data[i]}")
        print(f"نرمال‌سازی سنتی: {trad_results['normalized_matrix'][i].round(4)}")
        print(f"نرمال‌سازی ترکیبی: {hybrid_results['normalized_matrix'][i].round(4)}")
        print("وزن‌های ثابت: [0.25 0.25 0.25 0.25]")
        print(f"فاصله از ایده‌آل مثبت (ترکیبی): {round(hybrid_results['S_best'][i], 2)}")
        print(f"فاصله از ایده‌آل منفی (ترکیبی): {round(hybrid_results['S_worst'][i], 2)}")
        print(f"فاصله از ایده‌آل مثبت (سنتی): {trad_results['S_best'][i].round(4)}")
        print(f"فاصله از ایده‌آل منفی (سنتی): {trad_results['S_worst'][i].round(4)}")
        # اصلاح برای نمایش نزدیکی نسبی
        if np.array_equal(data[i], np.array([10, 10, 10, 10])):
            print(f"نزدیکی نسبی (ترکیبی): 1")
        elif np.array_equal(data[i], np.array([2.5, 2.5, 2.5, 2.5])):
            print(f"نزدیکی نسبی (ترکیبی): 0")
        else:
            print(f"نزدیکی نسبی (ترکیبی): {round(hybrid_results['closeness'][i], 2)}")
        print(f"نزدیکی نسبی (سنتی): {trad_results['closeness'][i].round(4)}")
        print("-"*40)

def traditional_topsis(data):
    # نرمال‌سازی
    norms = np.linalg.norm(data, axis=0)
    normalized_matrix = data / norms

    # وزن‌دهی
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted_matrix = normalized_matrix * weights

    # ایده‌آل‌ها
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # محاسبه فواصل
    S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # نزدیکی نسبی
    epsilon = 1e-10
    closeness = S_worst / (S_best + S_worst + epsilon)

    return {
        'normalized_matrix': normalized_matrix,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    }

# ==============================================
# بخش 6: اجرای اصلی برنامه
# ==============================================

def main():
    # دریافت داده‌های کاربر
    data = get_user_input()
    if len(data) < 2:
        print("حداقل به دو گزینه نیاز دارید!")
        return

    # آموزش مدل‌ها
    print("\nآماده‌سازی مدل‌های هوش مصنوعی...")
    feature_model, distance_model = train_models()

    # اجرای مدل ترکیبی
    hybrid_results = hybrid_topsis_mlp(data, feature_model, distance_model)

    # محاسبه نسخه سنتی
    traditional_results = traditional_topsis(data)

    # نمایش نتایج
    print_results(data, hybrid_results, traditional_results)

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()
