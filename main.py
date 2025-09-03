# main.py
import cv2
from segmentation import segment_image
from shape_detection import detect_casualties_and_pads
from assignment import assign_casualties

def process_image(image_path):
    # 1️⃣ Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # 2️⃣ Segment ocean and land
    segmented_img = segment_image(img)
    cv2.imwrite("output_segmented.jpg", segmented_img)

    # 3️⃣ Detect casualties & rescue pads
    casualties, pads = detect_casualties_and_pads(img)

    # 4️⃣ Assign casualties to pads
    assignments, pad_scores, rescue_ratio = assign_casualties(casualties, pads)

    # 5️⃣ Print results
    print(f"\nResults for {image_path}:")
    print("Assignments:", assignments)
    print("Pad Scores:", pad_scores)
    print("Rescue Ratio:", rescue_ratio)

    # 6️⃣ Save annotated image
    annotated_img = img.copy()
    for c in casualties:
        x, y, shape, color = c['x'], c['y'], c['shape'], c['color']
        cv2.putText(annotated_img, f"{shape}-{color}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.circle(annotated_img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite("output_annotated.jpg", annotated_img)

if __name__ == "__main__":
    # Example usage
    process_image("input_image.jpg")
