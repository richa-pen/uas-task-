# mai.py
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_DIR = "input_images/"
OUTPUT_DIR = "segmented_output/"
RESULTS_FILE = "results.csv"
PAD_CAPACITY = [4, 3, 2]  # blue, pink, grey
PAD_ORDER = ['blue','pink','grey']
AGE_PRIORITY = {'star':3, 'triangle':2, 'square':1}
EMERGENCY_PRIORITY = {'red':3, 'yellow':2, 'green':1}
CASUALTY_SHAPES = {10:'star', 3:'triangle', 4:'square'}

PAD_COLORS_BGR = {'blue': [255,0,0], 'pink': [180,105,255], 'grey': [160,160,160]}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def segment_land_ocean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ocean_mask = cv2.inRange(hsv, (90,50,50), (130,255,255))
    land_mask = cv2.inRange(hsv, (10,50,50), (40,255,255))
    segmented = img.copy()
    segmented[ocean_mask>0] = [255,0,0]
    segmented[land_mask>0]  = [34,139,34]
    return segmented

def detect_pads(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=15, maxRadius=35)
    pads = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles:
            x, y, r = c
            # Assign dummy ordering (blue/pink/grey) in code, could be improved with color matching
            pads.append({'coord': (x,y)})
    return pads

def detect_casualties(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    casualties = []
    color_ranges = {'red': ([0,70,50],[10,255,255]), 'yellow': ([25,70,70],[35,255,255]), 'green': ([40,40,40],[70,255,255])}
    for color, (low, high) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            shape = CASUALTY_SHAPES.get(len(approx), None)
            if not shape: continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            casualties.append({'shape': shape, 'color': color, 'coord': (cx, cy)})
    return casualties

def compute_distances(casualties, pads):
    dist_matrix = []
    for c in casualties:
        dists = []
        for p in pads:
            dx = c['coord'] - p['coord']
            dy = c['coord'][1] - p['coord'][1]
            dists.append(np.sqrt(dx**2 + dy**2))
        dist_matrix.append(dists)
    return dist_matrix

def assign_casualties(casualties, pads, dist_matrix):
    assignments = [[] for _ in PAD_CAPACITY]
    assigned = set()
    score_table = []
    for idx, casualty in enumerate(casualties):
        for i, pad in enumerate(pads):
            priority = AGE_PRIORITY[casualty['shape']] * EMERGENCY_PRIORITY[casualty['color']]
            score_table.append((priority, dist_matrix[idx][i], i, idx, casualty))
    score_table.sort(key=lambda x: (-x, x[1]))
    for entry in score_table:
        priority, dist, pad_idx, casualty_idx, casualty = entry
        if len(assignments[pad_idx]) < PAD_CAPACITY[pad_idx] and casualty_idx not in assigned:
            assignments[pad_idx].append(casualty)
            assigned.add(casualty_idx)
    return assignments

def main():
    img_names = []
    rescue_ratios = []
    results = []
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg') or f.endswith('.png')]

    for fname in input_files:
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        segmented = segment_land_ocean(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'seg_' + fname), segmented)

        pads = detect_pads(img)
        casualties = detect_casualties(img)
        dist_matrix = compute_distances(casualties, pads)
        assignments = assign_casualties(casualties, pads, dist_matrix)

        pad_priorities = []
        total_priority = 0
        for pad_casualties in assignments:
            pad_score = sum(AGE_PRIORITY[c['shape']] * EMERGENCY_PRIORITY[c['color']] for c in pad_casualties)
            pad_priorities.append(pad_score)
            total_priority += pad_score
        rescue_ratio = total_priority/len(casualties) if casualties else 0
        img_names.append(fname)
        rescue_ratios.append(rescue_ratio)

        results.append({
            'image': fname,
            'total_casualties': len(casualties),
            'shape_color_list': [f"{c['shape']}-{c['color']}" for c in casualties],
            'distance_matrix': dist_matrix,
            'assignments': [[f"[{AGE_PRIORITY[c['shape']]},{EMERGENCY_PRIORITY[c['color']]}]" for c in pad_casualties] for pad_casualties in assignments],
            'pad_priorities': pad_priorities,
            'rescue_ratio': rescue_ratio
        })

    ranks = sorted(zip(img_names, rescue_ratios), key=lambda x: -x[1])
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        f.write('image,total_casualties,shape_color_list,assignments,pad_priorities,rescue_ratio\n')
        for res in results:
            f.write(f"{res['image']},{res['total_casualties']},\"{res['shape_color_list']}\",\"{res['assignments']}\",{res['pad_priorities']},{res['rescue_ratio']}\n")
        f.write('\nRanked Images by Rescue Ratio:\n')
        for name, ratio in ranks:
            f.write(f"{name}: {ratio}\n")

    print("Processing complete.")
    print("Images ranked by rescue ratio (highest first):")
    for name, ratio in ranks:
        print(name, ratio)

if __name__ == '__main__':
    main()
