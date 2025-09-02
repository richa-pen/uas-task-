gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, 
                           param1=50, param2=30, minRadius=15, maxRadius=35)
pads = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles:
        pads.append({'coord': (c, c[1])})
        
        
Detect Casualties (Shapes and Colors)  
    
        # Loop through each color/emergency
casualties = []
colors = {'red': ([0,70,50],[10,255,255]), 'yellow': ([25,70,70],[35,255,255]), 'green': ([40,40,40],[70,255,255])}
for name, (low, high) in colors.items():
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        shape = None
        if len(approx) == 10:
            shape = 'star'
        elif len(approx) == 3:
            shape = 'triangle'
        elif len(approx) == 4:
            shape = 'square'
        else:
            continue
        M = cv2.moments(cnt)
        if M['m00']==0: continue
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        casualties.append({'shape': shape, 'color': name, 'coord': (cx, cy)})
