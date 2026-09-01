ENROLLMENT (once per registered student/staff):
for each person P with sample photos S:
    for each image I in S:
        gray = to_grayscale(I)
        gray = CLAHE(gray)                 # illumination correction
        box  = HaarCascade.detect(gray)    # face localisation
        face = resize(crop(gray, box), 120x120)
        add (face, label=P) to training_set
LBPH_model = train_LBPH(training_set)
 
GATE DECISION (per incoming frame):
gray = CLAHE(to_grayscale(frame))
for each box in HaarCascade.detect(gray):
    face = resize(crop(gray, box), 120x120)
    (label, distance) = LBPH_model.predict(face)
    if distance < THRESHOLD: ACCESS = GRANTED(label)
    else: ACCESS = DENIED(UNKNOWN)
    log(box, label, distance, ACCESS)
