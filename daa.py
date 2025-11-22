import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------
#  STRING MATCHING ALGORITHMS (Naive + KMP)
# ---------------------------------------------

def naive_search(text, pattern):
    positions = []
    N, M = len(text), len(pattern)
    for i in range(N - M + 1):
        match = True
        for j in range(M):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    return positions


def build_lps(pattern):
    lps = [0] * len(pattern)
    i = 1
    length = 0
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text, pattern):
    lps = build_lps(pattern)
    positions = []
    i = j = 0

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                positions.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return positions


# ---------------------------------------------
#  TEMPLATE MATCHING (OpenCV)
# ---------------------------------------------
def load_logo_templates(folder="logos"):
    templates = {}
    if not os.path.isdir(folder):
        print(f"Warning: templates folder '{folder}' not found. Logo detection will be disabled.\nCreate the folder and add template images (e.g. PNG/JPG) if you want detection.")
        return templates

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, 0)
        if img is None:
            continue
        name = os.path.splitext(file)[0]
        templates[name] = img

    if templates:
        print(f"Loaded {len(templates)} template(s): {', '.join(sorted(templates.keys()))}")
    else:
        print(f"No valid template images found in '{folder}'. Add PNG/JPG files there.")

    return templates


def detect_logos(frame, templates, threshold=0.7):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, temp in templates.items():
        # ensure template fits inside the frame; resize if template is larger
        img_h, img_w = gray.shape[:2]
        temp_h, temp_w = temp.shape[:2]

        temp_to_use = temp
        if temp_h > img_h or temp_w > img_w:
            # scale down to fit within image, keep a small margin
            scale = min(img_h / temp_h, img_w / temp_w) * 0.95
            new_h = max(1, int(temp_h * scale))
            new_w = max(1, int(temp_w * scale))
            temp_to_use = cv2.resize(temp, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = temp_to_use.shape[:2]
        result = cv2.matchTemplate(gray, temp_to_use, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # draw filled rectangle behind text for readability
            label = f"{name}: {max_val:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = top_left[0]
            text_y = max(0, top_left[1] - 12)
            cv2.rectangle(frame, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 4), (0, 255, 0), -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


# ---------------------------------------------
#  SPEED TEST FOR NAIVE vs KMP
# ---------------------------------------------
def compare_algorithms():
    text = "A" * 200000 + "B"
    pattern = "A" * 999 + "B"

    # Naive timing
    start = time.time()
    naive_search(text, pattern)
    naive_t = time.time() - start

    # KMP timing
    start = time.time()
    kmp_search(text, pattern)
    kmp_t = time.time() - start

    print("\n--- STRING MATCH ALGORITHM SPEED ---")
    print(f"Naive Search Time: {naive_t:.4f} sec")
    print(f"KMP Search Time:   {kmp_t:.4f} sec")

    # Plot
    plt.bar(["Naive", "KMP"], [naive_t, kmp_t])
    plt.title("Naive vs KMP Speed")
    plt.ylabel("Time (sec)")
    plt.show()


# ---------------------------------------------
#  PROCESS IMAGE FOLDER
# ---------------------------------------------
def process_image_folder(folder="images"):
    templates = load_logo_templates()

    if not templates:
        print("No logo templates found. Skipping image folder processing.")
        return
    if not os.path.isdir(folder):
        print(f"Warning: image folder '{folder}' not found. Create the folder and add images (PNG/JPG) to use this feature.")
        return

    # collect image files
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        print(f"No image files found in folder '{folder}'.")
        return

    for file in files:
        path = os.path.join(folder, file)
        frame = cv2.imread(path)
        if frame is None:
            continue

        frame = detect_logos(frame, templates)
        cv2.imshow("Detected Logos", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ---------------------------------------------
#  WEBCAM STREAM
# ---------------------------------------------
def webcam_mode():
    templates = load_logo_templates()

    if not templates:
        print("No logo templates found. Webcam mode cancelled.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not detected.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_logos(frame, templates)
            cv2.imshow("Webcam Logo Detection", frame)

            # Press 'q' in the window to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting webcam mode.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------
#  MAIN MENU
# ---------------------------------------------
def main():
    print("\n====== BRAND LOGO RECOGNITION PROJECT ======")
    print("1. Process Image Folder")
    print("2. Webcam Live Detection")
    print("3. Compare Naive vs KMP Speed")
    print("4. Exit")

    choice = input("\nEnter choice: ")

    # allow user to set the template matching threshold (helps tuning)
    try:
        thr_input = input("Enter matching threshold (0.0-1.0) [default 0.7]: ")
        threshold = float(thr_input) if thr_input.strip() != "" else 0.7
        if threshold < 0 or threshold > 1:
            print("Threshold out of range; using default 0.7")
            threshold = 0.7
    except Exception:
        threshold = 0.7

    if choice == "1":
        process_image_folder()
    elif choice == "2":
        webcam_mode()
    elif choice == "3":
        compare_algorithms()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()