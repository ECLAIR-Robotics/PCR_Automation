# main program!!
# beforehand: 
# fill beakers to 5 oz
# adjust arduino com in arduino IDE

# pseudocode in comments

# 1. CALIBRATE
# find focal length of camera

# LOOP THE FOLLOWING:

# 2. FIND APRIL TAG 0 AND 1
# function that takes in april tag ID

# 3. CALCULATE DISTANCE
# function that calculates distance between april tags

# 4. MOVE ARM TO APRIL TAG 1
# function that moves arm given set distance
# distance from arm to 0 is hard-coded

# 5. MAKE ARM GO DOWN [HARD-CODED MM]

# 6. CALL GET_LIQUID FROM HARDWARE

# 7. MAKE ARM GO UP [HARD-CODED MM]

# 8. FIND APRIL TAG 0 AND 2

# 9. CALCULATE DISTANCE

# 10. MOVE ARM TO APRIL TAG 2

# 11. MAKE ARM GO DOWN

# 12. CALL EJECT_LIQUID FROM HARDWARE

# 13. MAKE ARM GO UP

import sys
import os
import hardware
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from computer_vision.birds_eye_view.overhead2.birdseyeview import (
    set_up_birds_eye,
    get_x_and_y_for_tag,
)

def main():
    # Setting up birds eye view code
    tagSize = 0.061 # in meters
    camera_number = 0 
    offset_distance_cm = 7.4 # in centimeters
    offset_distance_m = offset_distance_cm / 100  # Convert to meters
    cap, detector, estimator, mtx = set_up_birds_eye(camera_number, tagSize)

    referenceX = 0
    referenceY = 0

    while True:
        # Move off to side first so we get a clear view of everything
        # INSERT CODE HERE

        # Get locations of two beakers
        x1, y1 = get_x_and_y_for_tag(cap, detector, estimator, offset_distance_m, mtx, 1, 20)
        x2, y2 = get_x_and_y_for_tag(cap, detector, estimator, offset_distance_m, mtx, 2, 20)
        if x1 is None or x2 is None:
            continue
        x1 = x1 + referenceX
        y1 = y1 + referenceY
        print(x1)
        print(y1)
        x2 = x2 + referenceX
        y2 = y2 + referenceY
        print(x2)
        print(y2)

        # Move to beaker 1 + move down
        # INSERT CODE HERE

        # Plunge down and intake liquid
        # INSERT CODE HERE
        hardware.get_liquid()

        # Move up from beaker
        # INSERT CODE HERE

        # Move to beaker 2 + move down
        # INSERT CODE HERE

        # Plunge down and eject liquid
        # INSERT CODE HERE
        hardware.eject_liquid()

        # Move up from beaker
        # INSERT CODE HERE


if __name__ == "__main__":
    main()