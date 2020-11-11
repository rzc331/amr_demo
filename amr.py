import pipette

#pipette.move_well(1, 0)

#pipette.move_tip(0)

pipette.measure(0)

pipette.home(1)
pipette.home(2)

pipette.get_tip(0)
pipette.get_liquid("solute", 0)
pipette.get_liquid("solvent", 100)
pipette.u1.set_3d_feeding(distance=-200, speed=30000, relative=True, timeout=60)
pipette.dispense(100, 3)
pipette.pipette("solute", 200, 7)
pipette.dispose_tip()
pipette.dispose_liq()
pipette.mixing(0, 1)
pipette.u2.send_cmd_sync("M2019")
pipette.u1.set_3d_feeding(distance=-200, relative=True, wait=False, timeout=60)
pipette.measure(0)

pipette.run(0.5, 0, 0)

import cv_digits
cv_digits.run(3)

cv_digits.take(3)

cv_digits.cv2.VideoCapture(0)
quit()

import active_learning

"/dev/ttyACM2(Arduino/Genuino Mega or Mega 2560)"
"/dev/ttyACM0"

