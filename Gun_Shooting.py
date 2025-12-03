"""
ME41006 Wild West Shooting 2025 Group 1
Green Target Detection and Auto-Shooting System
"""

import cv2
import numpy as np
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO

# ========== SERVO MOTOR SETUP ==========
# Initialize the PCA9685 servo controller
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

def set_servo_angle(channel, angle):
    """
    Set servo motor angle
    channel: servo channel (1=X-axis, 2=Y-axis, 3=trigger)
    angle: target angle in degrees
    """
    angle = 4096 * ((angle * 11) + 500) / 20000
    pwm.set_pwm(channel, 0, int(angle))

def shoot():
    """
    Trigger the gun
    """
    print("SHOOTING!")
    
    IN1 = 24
    IN2 = 23
    ENA = 18

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings (False)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)

    def motor_on():
        GPIO.output (IN1, GPIO.HIGH)
        GPIO.output (ENA, GPIO.HIGH)
        GPIO.output (IN2, GPIO.LOW)
        print("OPEN")
    def motor_off():
        GPIO.output (IN1, GPIO.LOW)
        GPIO.output (IN2, GPIO.LOW)
        GPIO.output (ENA, GPIO.LOW)
        print("CLOSE")
    try:
        while True:
            motor_on()
            time.sleep(0.7)
            motor_off()
            time.sleep(0.5)
            break
        

    except KeyboardInterrupt:
        print("over")
        GPIO.cleanup()

# Initialize servo positions to center
set_servo_angle(1, 90)  # X-axis center
set_servo_angle(2, 90)  # Y-axis center
set_servo_angle(3, 75)  # Trigger ready

# ========== CAMERA SETUP ==========
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# ========================================================================
# TUNABLE PARAMETERS - Adjust these values for your system
# ========================================================================

# --- Color Detection Settings ---
# Green color range in HSV (Hue, Saturation, Value)
# Adjust these to match your target's green color
green_lower = np.array([50, 80, 100])   # Lower bound [Hue, Saturation, Value]
green_upper = np.array([150, 255, 255]) # Upper bound [Hue, Saturation, Value]

# --- Target Size Filters ---
min_area = 1300   # Minimum pixel area to detect (ignore small objects)
max_area = 3000   # Maximum pixel area to detect (ignore large reflections/background)

# --- Region Masking ---
floor_ignore_percent = 0.30  # Ignore bottom 30% of frame (floor area)

# --- Tracking Control ---
targetCenter = [320, 240]  # Target aim point [X, Y] - adjust for shooting accuracy
Kp_x = 0.03  # Horizontal tracking speed (higher = faster, may oscillate)
Kp_y = 0.03  # Vertical tracking speed (higher = faster, may oscillate)

# --- Lock Detection ---
deadzone_x = 30  # Horizontal tolerance in pixels before shooting
deadzone_y = 30  # Vertical tolerance in pixels before shooting

# --- Shooting Parameters ---
shot_cooldown = 2.0          # Seconds between shots
aim_adjustment = 1.0         # Degrees to aim down before shooting (compensate for trajectory)
aim_stabilize_time = 0.1     # Seconds to wait after aiming before shooting
shooting_angle_offset = -14.0  # Y-axis offset after lock (+ aims down, - aims up) to compensate if gun shoots high/low

# --- Idle Behavior (when no target detected) ---
idle_mode = "none"        # "scan" = sweep left/right, "reset" = return to center (90, 90, 75), "none" = do nothing
idle_scan_min = 40        # Minimum scan angle (degrees) - used in "scan" mode
idle_scan_max = 140       # Maximum scan angle (degrees) - used in "scan" mode
idle_scan_speed = 3       # Scan speed (degrees per frame) - used in "scan" mode

# --- Servo Angle Limits ---
servo_angle_min = 30   # Minimum safe servo angle
servo_angle_max = 150  # Maximum safe servo angle

# ========================================================================
# END TUNABLE PARAMETERS
# ========================================================================

# Internal variables - do not modify
currentAngle_x = 90
currentAngle_y = 90
last_shot_time = 0
idle_scan_direction = 1  # 1 for right, -1 for left
idle_angle_x = 90  # Current angle during idle scan

print("=" * 50)
print("GREEN TARGET TRACKING AND AUTO-SHOOTING SYSTEM")
print("=" * 50)
print("Press 'q' to quit")
print("Press 's' to save current frame")
print("System starting...")
print("=" * 50)

time.sleep(1)

# ========== MAIN TRACKING LOOP ==========
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Draw crosshair on frame
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)
        
        original = frame.copy()
        
        # Pre-processing
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color
        mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Create region mask to ignore floor area
        region_mask = np.ones(mask.shape, dtype=np.uint8) * 255
        floor_cutoff = int(h * (1.0 - floor_ignore_percent))
        region_mask[floor_cutoff:, :] = 0
        
        # Apply region mask to color mask
        mask = cv2.bitwise_and(mask, region_mask)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Apply mask
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find contours
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        area = 1
        target_found = False
        
        # Loop through the contours
        for c in contours:
            area = cv2.contourArea(c)
            
            # Check if area is within valid range (not too small, not too large)
            if min_area < area < max_area:
                target_found = True
                print("area:", area)
                
                # Draw contour
                cv2.drawContours(res, [c], contourIdx=-1, color=(255, 255, 255), 
                               thickness=5, lineType=cv2.LINE_AA)
                
                # Calculate the center of the object
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(res, (cx, cy), 5, (255, 0, 0), -1)
                print("center:", cx, cy)
                
                # Calculate error from center
                error_x = cx - targetCenter[0]
                error_y = cy - targetCenter[1]
                
                # Check if target is centered (within deadzone)
                is_locked = abs(error_x) < deadzone_x and abs(error_y) < deadzone_y
                
                # Only move servos if NOT locked (outside deadzone)
                if not is_locked:
                    # Proportional control - calculate new angles
                    new_angle_x = currentAngle_x - (Kp_x * error_x)
                    new_angle_y = currentAngle_y - (Kp_y * error_y)
                    
                    # Limit angles to safe range
                    new_angle_x = max(servo_angle_min, min(servo_angle_max, new_angle_x))
                    new_angle_y = max(servo_angle_min, min(servo_angle_max, new_angle_y))
                    
                    # Update servo positions
                    set_servo_angle(1, new_angle_x)
                    set_servo_angle(2, new_angle_y)
                    
                    # Update current angles
                    currentAngle_x = new_angle_x
                    currentAngle_y = new_angle_y
                
                # Display lock status
                if is_locked:
                    cv2.putText(res, "TARGET LOCKED", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Check cooldown before shooting
                    current_time = time.time()
                    if current_time - last_shot_time > shot_cooldown:
                        # Apply shooting angle offset and aim adjustment
                        shooting_angle_y = currentAngle_y + shooting_angle_offset + aim_adjustment
                        set_servo_angle(2, shooting_angle_y)
                        # time.sleep(aim_stabilize_time)
                        
                        shoot()
                        
                        # Return to locked position
                        set_servo_angle(2, currentAngle_y)
                        
                        last_shot_time = current_time
                else:
                    cv2.putText(res, "TRACKING...", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Display servo angles
                angle_text = "Angles: X={:.1f} Y={:.1f}".format(
                    currentAngle_x, currentAngle_y)
                cv2.putText(res, angle_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                print("Target at ({}, {}) | Error: ({:.1f}, {:.1f}) | Angles: ({:.1f}, {:.1f})".format(
                    cx, cy, error_x, error_y, currentAngle_x, currentAngle_y))
                
                # Only track the largest target
                break
        
        if not target_found:
            if idle_mode == "scan":
                cv2.putText(res, "NO TARGET - SCANNING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Idle scanning mode - sweep left and right
                idle_angle_x += idle_scan_speed * idle_scan_direction
                
                # Change direction when reaching limits
                if idle_angle_x >= idle_scan_max:
                    idle_angle_x = idle_scan_max
                    idle_scan_direction = -1
                elif idle_angle_x <= idle_scan_min:
                    idle_angle_x = idle_scan_min
                    idle_scan_direction = 1
                
                # Move servo to scanning position
                set_servo_angle(1, idle_angle_x)
                currentAngle_x = idle_angle_x
                
                # Keep Y-axis level (horizontal to ground)
                set_servo_angle(2, 90)
                currentAngle_y = 90
                
                # Display scanning angle
                scan_text = "Scanning: X={:.1f}".format(idle_angle_x)
                cv2.putText(res, scan_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            elif idle_mode == "reset":
                cv2.putText(res, "NO TARGET", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Reset all motors to center position
                set_servo_angle(1, 90)
                set_servo_angle(2, 90)
                set_servo_angle(3, 75)
            
            else:  # idle_mode == "none" or any other value
                cv2.putText(res, "NO TARGET", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Do nothing - servos stay at last position
                currentAngle_x = 90
                currentAngle_y = 90
        
        # Display frames
        cv2.imshow("Original", original)
        cv2.imshow("Mask", mask)
        cv2.imshow("Target Tracking", res)
        
        # Handle keyboard input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Exiting program...")
            break
        elif k == ord('s'):
            # Save current frames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite('original_{}.jpg'.format(timestamp), original)
            cv2.imwrite('mask_{}.jpg'.format(timestamp), mask)
            cv2.imwrite('result_{}.jpg'.format(timestamp), res)
            print("Frames saved with timestamp: {}".format(timestamp))

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    # Cleanup
    print("Resetting servos to center position...")
    set_servo_angle(1, 90)
    set_servo_angle(2, 90)
    set_servo_angle(3, 75)
    
    cap.release()
    cv2.destroyAllWindows()
    print("System shutdown complete.")
