from ultralytics import YOLO
import cv2
import time
import numpy as np

MODEL_PATH = "/Users/khadijahbaothman/Desktop/egg counter/runs/detect/egg_model2/weights/best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

ROI_X1, ROI_Y1 = 180, 140
ROI_X2, ROI_Y2 = 980, 620

TARGET = 6
CONF = 0.6

# tracking 345
plate_present = False
locked_count = 0

total_plates = 0
pass_count = 0
fail_count = 0

last_result = "--"

missing_frames = 0

stable_counter = 0
last_seen = None

STABLE_REQUIRED = 5
LOCK_DELAY_SEC = 0.6
first_detect_time = None

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    results = model.predict(roi, conf=CONF, verbose=False)

    current_count = len(results[0].boxes)

    # رسم ROI
    cv2.rectangle(frame,(ROI_X1,ROI_Y1),(ROI_X2,ROI_Y2),(0,255,0),2)

    # رسم boxes
    for b in results[0].boxes:
        x1,y1,x2,y2 = map(int,b.xyxy[0])
        cv2.rectangle(frame,
                      (x1+ROI_X1,y1+ROI_Y1),
                      (x2+ROI_X1,y2+ROI_Y1),
                      (0,0,255),2)

    # ===== Logic with delay =====

    if current_count > 0:

        missing_frames = 0

        if not plate_present:

            if last_seen == current_count:
                stable_counter += 1
            else:
                stable_counter = 1
                last_seen = current_count
                first_detect_time = time.time()

            delay_passed = (time.time() - first_detect_time) >= LOCK_DELAY_SEC

            if stable_counter >= STABLE_REQUIRED and delay_passed:

                locked_count = current_count
                plate_present = True

                total_plates += 1

                if locked_count >= TARGET:

                    pass_count += 1
                    last_result = "PASS"

                else:

                    fail_count += 1
                    last_result = "FAIL"

                print(f"Plate {total_plates} → {locked_count} eggs → {last_result}")

    else:

        if plate_present:

            missing_frames += 1

            if missing_frames >= 8:

                plate_present = False

        else:

            stable_counter = 0
            last_seen = None

    # ===== Dashboard Panel =====

    panel_w = 380
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    # title
    cv2.putText(panel,"EGG COUNTER",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.line(panel,(20,60),(panel_w-20,60),(120,120,120),2)

    # status
    status = "LOCKED" if plate_present else "READY"
    color = (0,255,0) if plate_present else (0,200,255)

    cv2.circle(panel,(30,100),10,color,-1)

    cv2.putText(panel,f"STATUS: {status}",(60,105),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # eggs
    cv2.putText(panel,"EGGS",(20,160),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),2)

    cv2.putText(panel,f"{locked_count} / {TARGET}",(20,240),
                cv2.FONT_HERSHEY_SIMPLEX,2.2,(0,0,255),4)

    # last result
    result_color = (0,255,0) if last_result=="PASS" else (0,0,255)

    cv2.putText(panel,f"LAST: {last_result}",(20,300),
                cv2.FONT_HERSHEY_SIMPLEX,1,result_color,3)

    # stats
    cv2.putText(panel,f"TOTAL: {total_plates}",(20,360),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(panel,f"PASS: {pass_count}",(20,400),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(panel,f"FAIL: {fail_count}",(20,440),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    pass_rate = (pass_count/total_plates*100) if total_plates>0 else 0

    cv2.putText(panel,f"PASS RATE: {pass_rate:.1f}%",(20,480),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(panel,f"LIVE: {current_count}",(20,520),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

    combined = np.hstack((frame, panel))

    cv2.imshow("Egg Counter Dashboard",combined)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == ord('r'):
        plate_present=False

cap.release()
cv2.destroyAllWindows()