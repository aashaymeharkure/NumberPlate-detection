import cv2
import pytesseract
import sys
import os
import csv
from datetime import datetime

# ─────────────────────────────────────────────
#  EXISTING FUNCTION — unchanged
#  Called by Java with: python extractor.py <image_path>
# ─────────────────────────────────────────────
def extract_plate_number(image_path):
    if not os.path.exists(image_path):
        print("ERROR: Image file not found at path.")
        return
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        text = pytesseract.image_to_string(binary, config=custom_config)
        cleaned_text = "".join(text.split()).strip()
        print(cleaned_text)
    except Exception as e:
        print(f"ERROR: An exception occurred in Python script: {e}")


# ─────────────────────────────────────────────
#  NEW FUNCTION — Webcam Detection Mode
#  Called by Java with: python extractor.py --webcam
#  Or run directly:     python extractor.py --webcam
# ─────────────────────────────────────────────
def run_webcam_detection():
    # Save detections log into fastDatabase folder (same as violations.dat)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "fastDatabase", "detections.csv")
    log_path = os.path.normpath(log_path)

    # Open webcam (0 = built-in, change to 1 for external USB camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Prepare CSV log file
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if not file_exists:
        writer.writerow(["Plate Text", "Timestamp"])

    print("Webcam started. Press 'q' to quit, 's' to save a snapshot.")

    # Tesseract config — same whitelist as image mode
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    # Haar cascade for plate region detection (built into OpenCV, no extra download)
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    plate_cascade = cv2.CascadeClassifier(cascade_path)

    last_detected = ""  # avoid logging the same plate repeatedly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detect plate regions using Haar cascade
        plates = plate_cascade.detectMultiScale(
            gray_filtered,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 20)
        )

        for (x, y, w, h) in plates:
            # Draw green bounding box around detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop and preprocess the plate region for OCR
            plate_roi = gray_filtered[y:y + h, x:x + w]
            binary = cv2.adaptiveThreshold(
                plate_roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Run Tesseract OCR on the plate
            text = pytesseract.image_to_string(binary, config=custom_config)
            cleaned = "".join(text.split()).strip()

            if len(cleaned) >= 4:  # ignore very short / noisy reads
                # Display plate text above bounding box
                cv2.putText(frame, cleaned, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Log to CSV only when a new plate is detected
                if cleaned != last_detected:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([cleaned, timestamp])
                    log_file.flush()
                    print(f"DETECTED: {cleaned} at {timestamp}")
                    last_detected = cleaned

        # Show hint text on the live window
        cv2.putText(frame, "Press Q to quit | S to snapshot", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("ANPR - Webcam Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Q — stop webcam
        if key == ord("q"):
            print("Webcam detection stopped.")
            break

        # S — save a snapshot of the current frame into the Img folder
        if key == ord("s"):
            snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            snap_path = os.path.join(script_dir, "..", "Img", snap_name)
            snap_path = os.path.normpath(snap_path)
            cv2.imwrite(snap_path, frame)
            print(f"Snapshot saved: {snap_path}")

    cap.release()
    log_file.close()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--webcam":
            # New webcam mode: python extractor.py --webcam
            run_webcam_detection()
        else:
            # Existing image mode: python extractor.py <image_path>
            extract_plate_number(sys.argv[1])
    else:
        print("ERROR: No argument provided.")
        print("Usage:")
        print("  Image mode:  python extractor.py <image_path>")
        print("  Webcam mode: python extractor.py --webcam")
