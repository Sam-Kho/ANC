
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' to avoid graphical issues


# وارد کردن کتابخانه‌ها
import librosa  # برای بارگذاری فایل‌های صوتی و پردازش سیگنال‌های صوتی
import numpy as np  # برای انجام محاسبات عددی مانند تولید نویز
import matplotlib.pyplot as plt  # برای ترسیم نمودارهای گرافیکی
import scipy.signal as signal  # برای استفاده از توابع پردازش سیگنال مانند فیلتر پایین‌گذر
import soundfile as sf  # برای ذخیره سیگنال‌ها به فایل‌های صوتی
import sounddevice as sd  # برای پخش صدا

# تابع برای اعمال فیلتر پایین‌گذر
# این فیلتر فرکانس‌های بالای یک حد مشخص (cutoff) را حذف می‌کند تا نویزهای با فرکانس بالا را کاهش دهد
def low_pass_filter(audio, sr, cutoff=5000):
    # محاسبه فرکانس نیویوک (Nyquist) که برابر با نصف نرخ نمونه‌برداری است
    nyquist = 0.5 * sr
    # نرمال‌سازی فرکانس cutoff نسبت به فرکانس نیویوک
    normal_cutoff = cutoff / nyquist
    # طراحی فیلتر پایین‌گذر با استفاده از تابع butter از scipy
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    # اعمال فیلتر پایین‌گذر به سیگنال
    return signal.filtfilt(b, a, audio)

# تابع برای اجرای الگوریتم NLMS (Normalized Least Mean Squares)
# این الگوریتم یک فیلتر تطبیقی است که سعی می‌کند نویز موجود در سیگنال را حذف کند
def nlms_filter(desired, input_signal, mu=0.001, M=128, epsilon=1e-6):
    N = len(desired)  # تعداد نمونه‌ها
    w = np.zeros(M)  # وزن‌های فیلتر تطبیقی (مقدار اولیه صفر)
    output_signal = np.zeros(N)  # سیگنال خروجی که نویز آن کمتر شده
    error_signal = np.zeros(N)   # سیگنال خطا (اختلاف بین سیگنال مطلوب و سیگنال خروجی)

    # حلقه برای پردازش سیگنال در هر نمونه
    for n in range(M, N):
        # گرفتن پنجره از ورودی که طول آن برابر با M است
        x = input_signal[n-M:n]
        # محاسبه سیگنال خروجی با ضرب نقطه‌ای وزن‌ها و ورودی
        output_signal[n] = np.dot(w, x)
        # محاسبه خطا به‌عنوان تفاوت بین سیگنال مطلوب و سیگنال خروجی
        error_signal[n] = desired[n] - output_signal[n]

        # محاسبه نرخ یادگیری نرمال‌شده که به مقدار ورودی بستگی دارد
        mu_n = mu / (np.dot(x, x) + epsilon)
        # به‌روزرسانی وزن‌های فیلتر تطبیقی
        w = w + mu_n * error_signal[n] * x

    # بازگشت سیگنال خروجی و سیگنال خطا
    return output_signal, error_signal

# بارگذاری یک سیگنال صوتی نمونه از کتابخانه librosa
audio, sr = librosa.load(librosa.example('trumpet'), sr=None)
# افزودن نویز به سیگنال صوتی با سطح نویز مشخص
noise_level = 0.1
noisy_audio = audio + noise_level * np.random.randn(len(audio))

# پیش‌پردازش سیگنال نویزی: اعمال فیلتر پایین‌گذر برای کاهش نویزهای با فرکانس بالا
noisy_audio_filtered = low_pass_filter(noisy_audio, sr, cutoff=5000)

# تنظیمات اولیه برای الگوریتم NLMS:
mu = 0.002  # نرخ یادگیری که بر همگرایی الگوریتم تأثیر دارد
M = 128     # طول فیلتر تطبیقی، این مقدار نشان‌دهنده تعداد نمونه‌های ورودی است که برای فیلتر استفاده می‌شود

# اجرای الگوریتم NLMS برای کاهش نویز در سیگنال صوتی
output_audio, error_audio = nlms_filter(audio, noisy_audio_filtered, mu=mu, M=M)

# ذخیره سیگنال‌های پردازش‌شده و خطا به فایل‌های صوتی
sf.write('output_audio_nlms_improved.wav', output_audio, sr)
sf.write('error_audio_nlms_improved.wav', error_audio, sr)

# پخش سیگنال خروجی پردازش‌شده با استفاده از کتابخانه sounddevice
sd.play(output_audio, sr)
sd.wait()  # منتظر می‌مانیم تا پخش به اتمام برسد

# نمایش گراف‌های سیگنال‌های مختلف برای مقایسه
plt.figure(figsize=(10, 6))

# نمایش سیگنال اصلی
plt.subplot(3, 1, 1)
plt.plot(audio[:5000])
plt.title('Original Audio')

# نمایش سیگنال نویزی
plt.subplot(3, 1, 2)
plt.plot(noisy_audio[:5000])
plt.title('Noisy Audio')

# نمایش سیگنال خروجی پس از کاهش نویز با الگوریتم NLMS
plt.subplot(3, 1, 3)
plt.plot(output_audio[:5000])
plt.title('Output Audio (Improved NLMS)')

plt.tight_layout()
plt.show()
