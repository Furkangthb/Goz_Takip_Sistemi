pip install opencv-python dlib numpy

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_center_y(eye_points, facial_landmarks):
    """Göz kapaklarının dikey merkezini hesaplar (Piksel hassasiyetinde)."""
    y_coords = [facial_landmarks.part(point).y for point in eye_points]
    return sum(y_coords) / len(y_coords)

lines = [
    "1. SATIR: Goruntu isleme projesine hos geldin.",
    "2. SATIR: Goz kapaklarinin hareketini takip ediyoruz.",
    "3. SATIR: Artik 239 sayisi sabit kalmayacak!",
    "4. SATIR: 'q' tusuna basarak cikis yapabilirsin."
]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Okuma Takibi")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Varsayılan metinleri çiz
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (30, 100 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        for face in faces:
            landmarks = predictor(gray, face)
            
            # Sol göz (36-41) ve Sağ göz (42-47) landmark y-ortalamaları
            left_y = get_eye_center_y(range(36, 42), landmarks)
            right_y = get_eye_center_y(range(42, 48), landmarks)
            avg_y = (left_y + right_y) / 2
            
            # --- KALİBRASYON TESTİ ---
            # Ekranda Goz Y'nin artik 238.4, 239.1 gibi degistigini goreceksin.
            # Eger degisim cok azsa (orn: 238-240 arasi), su araligi daralt:
            active_line = int(np.interp(avg_y, [150, 165], [0, len(lines)-1]))
            active_line = max(0, min(active_line, len(lines)-1))

            # Aktif satırı boya
            cv2.putText(frame, lines[active_line], (30, 100 + active_line*60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Verileri ekrana bas
            cv2.circle(frame, (20, int(avg_y)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Goz Y: {avg_y:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow("Okuma Takibi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)

import cv2
import dlib
import numpy as np
from collections import deque

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_center_y(eye_points, facial_landmarks):
    y_coords = [facial_landmarks.part(point).y for point in eye_points]
    return sum(y_coords) / len(y_coords)

# --- YUMUŞATMA AYARI ---
# Son 15 karenin ortalamasını alacağız (Değer arttıkça gecikme artar ama titreme azalır)
smoothing_window = 15
history_y = deque(maxlen=smoothing_window)

lines = [
    "1. SATIR: Goruntu isleme projesine hos geldin.",
    "2. SATIR: Yumusatma filtresi aktif edildi.",
    "3. SATIR: Artik degerler cok daha stabil.",
    "4. SATIR: 'q' tusuna basarak cikis yapabilirsin."
]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Okuma Takibi")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Arka plana metinleri çiz
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (30, 100 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        for face in faces:
            landmarks = predictor(gray, face)
            
            left_y = get_eye_center_y(range(36, 42), landmarks)
            right_y = get_eye_center_y(range(42, 48), landmarks)
            raw_avg_y = (left_y + right_y) / 2
            
            # --- FİLTRE UYGULAMA ---
            history_y.append(raw_avg_y)
            smooth_avg_y = sum(history_y) / len(history_y)
            
            # Kalibrasyon (Kendi değerlerine göre burayı güncelle: [Min_Y, Max_Y])
            # Sayılar çok oynuyorsa bu aralığı biraz genişletebilirsin (Örn: [235, 245])
            active_line = int(np.interp(smooth_avg_y, [170, 190], [0, len(lines)-1]))
            active_line = max(0, min(active_line, len(lines)-1))

            # Aktif satırı boya
            cv2.putText(frame, lines[active_line], (30, 100 + active_line*60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Verileri ekrana bas
            cv2.circle(frame, (20, int(smooth_avg_y)), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"Smooth Y: {smooth_avg_y:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow("Okuma Takibi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)

import cv2
import mediapipe as mp
import numpy as np

# =========================
# AYARLAR & SABİTLER
# =========================
WIDTH, HEIGHT = 1280, 720
FONT_SCALE = 1.2
LINE_HEIGHT = 80
START_Y = 150  # Metnin başladığı Y koordinatı

# Kafa hareketi telafisi (Kafa aşağı eğilince gözün kaymasını engeller)
# 0.25 - 0.35 arası idealdir.
HEAD_COMPENSATION = 0.30 

text_lines = [
    "1. SATIR: Goruntu isleme projesine hos geldin.",
    "2. SATIR: Su an MediaPipe ve Kalman Filtresi calisiyor.",
    "3. SATIR: Gozlerinle okudugun satiri algiliyorum.",
    "4. SATIR: Kafani hafifce oynatsan bile takip bozulmaz.",
    "5. SATIR: Cikis yapmak icin 'ESC' tusuna basabilirsin."
]

# =========================
# MEDIAPIPE KURULUMU
# =========================

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# KALMAN FİLTRESİ (Titreme Önleyici)
# =========================
kalman = cv2.KalmanFilter(2, 1)
kalman.measurementMatrix = np.array([[1, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01  # Hassasiyet ayarı (Düşük = Daha az titreme)

# =========================
# DEĞİŞKENLER
# =========================
state = "CALIB"  # Başlangıç durumu: Kalibrasyon
calib_step = 0
calib_samples = []  # O anki satır için toplanan veriler
calib_data = []     # Tüm satırların ortalamaları

# Affine dönüşüm katsayıları (Göz Y -> Ekran Y)
slope = 1.0
intercept = 0.0

cap = cv2.VideoCapture(0)

print("Sistem Başlatılıyor... Lütfen kameraya bak.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aynalama ve Boyutlandırma
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Yüz İşleme
    results = face_mesh.process(rgb)
    
    raw_gaze_y = None
    filtered_gaze_y = 0

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        
        # Landmark Noktaları (İris merkezleri ve Burun ucu)
        left_eye = face.landmark[468]
        right_eye = face.landmark[473]
        nose = face.landmark[1]

        # Gözlerin Y ortalaması
        eye_y_avg = (left_eye.y + right_eye.y) / 2
        
        # Ham Göz Verisi (Kafa telafisi ile)
        # Formül: Göz_Y - (Katsayı * Burun_Y)
        raw_gaze = eye_y_avg - (HEAD_COMPENSATION * nose.y)
        
        # Kalman Filtresi Uygula
        kalman.correct(np.array([[np.float32(raw_gaze)]]))
        prediction = kalman.predict()
        filtered_gaze_y = prediction[0][0]

    # =========================
    # ARAYÜZ VE MANTIK
    # =========================
    
    # --- DURUM 1: KALİBRASYON ---
    if state == "CALIB":
        # Kullanıcıya nereye bakması gerektiğini göster
        target_y = START_Y + (calib_step * LINE_HEIGHT)
        
        cv2.putText(frame, "LUTFEN BU SATIRA BAKIN (Kalibrasyon)", (150, target_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # İlerleme çubuğu mantığı
        if results.multi_face_landmarks:
            calib_samples.append(filtered_gaze_y)
            
            # Yeterli veri toplandı mı? (30 frame yaklaşık 1 saniye)
            if len(calib_samples) > 30:
                mean_val = np.mean(calib_samples)
                calib_data.append(mean_val)
                print(f"Satır {calib_step+1} kalibre edildi. Değer: {mean_val:.4f}")
                
                calib_samples = []
                calib_step += 1
                
                # Tüm satırlar bitti mi?
                if calib_step >= len(text_lines):
                    # Lineer Regresyon (Polyfit) ile dönüşüm katsayılarını hesapla
                    # Göz verileri (X) -> Ekran Y koordinatları (Y)
                    screen_y_coords = [START_Y + i * LINE_HEIGHT for i in range(len(text_lines))]
                    slope, intercept = np.polyfit(calib_data, screen_y_coords, 1)
                    state = "READ"
                    print("Kalibrasyon Tamamlandı! Okuma Moduna Geçiliyor.")

    # --- DURUM 2: OKUMA MODU ---
    elif state == "READ":
        if results.multi_face_landmarks:
            # Göz değerini ekran koordinatına dönüştür: y = mx + b
            screen_gaze_y = int(slope * filtered_gaze_y + intercept)
            
            # Hangi satırda olduğunu bul
            # (screen_gaze_y - Başlangıç) / Satır Yüksekliği
            line_index = int((screen_gaze_y - (START_Y - LINE_HEIGHT/2)) / LINE_HEIGHT)
            
            # Sınırlandırma (0 ile max satır arası)
            line_index = max(0, min(line_index, len(text_lines) - 1))
            
            # --- ÇİZİM İŞLEMLERİ ---
            
            # 1. Metinleri Yazdır ve Aktif Satırı Boya
            for i, line in enumerate(text_lines):
                y_pos = START_Y + i * LINE_HEIGHT
                
                # Eğer bu satır okunuyorsa arka planı boya (Highlight)
                if i == line_index:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (50, y_pos - 60), (WIDTH - 50, y_pos + 20), (0, 255, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0) # Şeffaflık
                
                color = (0, 0, 0) if i == line_index else (200, 200, 200)
                cv2.putText(frame, line, (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2)

            # 2. Göz İmleci (Kırmızı Nokta) - Nereye baktığını görmek için
            cv2.circle(frame, (WIDTH//2, screen_gaze_y), 10, (0, 0, 255), -1)
            cv2.line(frame, (0, screen_gaze_y), (WIDTH, screen_gaze_y), (0, 0, 255), 1)

    # Bilgi Paneli
    cv2.putText(frame, "Cikis: ESC | Yeniden Kalibrasyon: 'R'", (20, 40), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Hibrit Goz Takip Sistemi", frame)
    
    key = cv2.waitKey(1)
    if key == 27: # ESC
        break
    elif key == ord('r'): # Yeniden Başlat
        state = "CALIB"
        calib_step = 0
        calib_samples = []
        calib_data = []

cap.release()
cv2.destroyAllWindows()



