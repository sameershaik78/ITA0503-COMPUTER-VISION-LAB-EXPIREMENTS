CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
 
def enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)
 
def detect_faces(gray_enhanced):
    return face_cascade.detectMultiScale(
        gray_enhanced, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
 
def build_enrollment(enroll_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8)
    samples, labels, label_map = [], [], {}
    for idx, person_dir in enumerate(sorted(glob(enroll_dir + '/enroll_*'))):
        label_map[idx] = basename(person_dir).replace('enroll_', '')
        for f in glob(person_dir + '/*.jpg'):
            gray = enhance(imread_gray(f))
            box = detect_faces(gray)[0]
            samples.append(resize(crop(gray, box), (120, 120)))
            labels.append(idx)
    recognizer.train(samples, np.array(labels))
    return recognizer, label_map
 
def run_gate(image_path, recognizer, label_map, threshold=70.0):
    gray = enhance(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY))
    for box in detect_faces(gray):
        face = resize(crop(gray, box), (120, 120))
        label, confidence = recognizer.predict(face)   # LBPH distance
        name = label_map[label] if confidence < threshold else 'UNKNOWN'
        yield box, name, confidence, ('GRANTED' if name != 'UNKNOWN' else 'DENIED')
