import sys, os
sys.path.append('/home/li/Documents/active_learning/AMR')
from time import sleep
from uarm.wrapper import SwiftAPI
import cv_digits


# load coords
well_coords = []
tip_coords = []

with open("coords/well_coords.txt", 'r') as f:
    for line in f:
        x, y = line.strip().split(",")
        well_coords.append((float(x), float(y)))

with open("coords/tip_coords.txt", 'r') as f:
    for line in f:
        x, y = line.strip().split(",")
        tip_coords.append((float(x), float(y)))

# connect
u1 = SwiftAPI(port="/dev/ttyACM0")
u2 = SwiftAPI(port="/dev/ttyACM1")
sleep(3)

# allowing extrusion
u1.send_cmd_sync("M302 S0")

# adjust end effector offset
u1.send_cmd_sync("M2411 S94")
u2.send_cmd_sync("M2411 S76")

# set speed factor for u1
u2.set_speed_factor(0.3)

#parameter setting
sip_more = 10
sip_offset = 15
spd = 30000
mixing_times = 2
total_vol = 300
tip_max_vol = 200
row = 4

# u1 coords
solute_x, solute_y, solute_z = 290, -144, 0
solvent_x, solvent_y, solvent_z = 349, -104, 1
well_starting_x, well_starting_y, well_starting_z = 265 + row * 9, -80, 2
tip_starting_x, tip_starting_y, tip_starting_z = 171, -218, -2
trash_x, trash_y, trash_z = 121, -172, 50
trash_liq_x, trash_liq_y, trash_liq_z = 211, -51, 5
home_x, home_y, home_z = 175, 0, 90
max_z = 75

# u2 coords
well_starting_u2 = [302.5, -62 + row * 9, -24.3]
well_final_u2 = [257, -62.5 + row * 9, -24.54]
total_count = 5
ultrasonic_1 = [1, -280, 160]
ultrasonic_2 = [66, -280, 160]
fan_x, fan_y, fan_z = 158, -133, 100
home_x_u2, home_y_u2, home_z_u2 = 180, 0, 180
max_z_u2 = 200


def run(conc, well_counter, tip_counter):
    solute_vol = total_vol * conc
    solvent_vol = total_vol * (1 - conc)

    # homing
    home(1)
    sleep(2)

    # acquire pipette tip
    get_tip(count=tip_counter)

    # get solvent and dispense
    pipette('solvent', solvent_vol, well_counter)
    dispose_tip()

    # acquire another pipette tip
    get_tip(count=tip_counter + 1)

    # get solute and dispense
    pipette('solute', solute_vol, well_counter)
    dispose_tip()

    # acquire another pipette tip
    get_tip(count=tip_counter + 2)

    # mixing
    mixing(well_counter, mixing_times)
    dispose_tip()

    # u1 homing
    home(1)

    # measure
    camera_id = cv_digits.get_camera_id()
    meter_cleaned(camera_id)  # proceed only when meter is cleaned
    measure(well_counter, camera_id)
    sleep(10)
    y_observed = cv_digits.run(camera_id)

    # clean and dry sensor
    clean()
    sleep(10)
    fan()

    return y_observed


def home(u):
    if u == 1:
        u1.set_wrist(90)
        u1.set_position(z=max_z, speed=spd, wait=False, timeout=10)
        u1.set_position(x=home_x, y=home_y, speed=spd, wait=False, timeout=10)
    elif u == 2:
        u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
        u2.set_position(x=home_x_u2, y=home_y_u2, speed=spd, wait=False, timeout=10)
    else:
        raise Exception("Please enter 1 or 2", u)


def get_tip(count):
    tip_x, tip_y = tip_coords[count]
    u1.set_position(z=max_z, speed=spd, wait=True, timeout=10)
    u1.set_position(tip_starting_x + tip_x, tip_starting_y + tip_y, speed=spd, wait=True, timeout=10)
    # u1.set_position(z=tip_starting_z + 10, speed=spd, wait=True, timeout=10)
    u1.set_position(z=tip_starting_z + 3, speed=spd / 3, wait=True, timeout=10)
    u1.set_position(z=tip_starting_z, speed=spd / 100, wait=True, timeout=10)
    sleep(1)
    # u1.set_position(z=tip_starting_z - 2, speed=spd / 100, wait=True, timeout=10)
    u1.set_position(z=tip_starting_z + 3, speed=spd / 10, wait=True, timeout=10)
    u1.set_position(z=tip_starting_z + 70, speed=spd, wait=True, timeout=10)


def move_tip(count):
    tip_x, tip_y = tip_coords[count]
    u1.set_position(z=max_z, speed=spd, wait=True, timeout=10)
    u1.set_position(tip_starting_x + tip_x, tip_starting_y + tip_y, speed=spd, wait=True, timeout=10)
    # u1.set_position(z=tip_starting_z + 10, speed=spd, wait=True, timeout=10)
    u1.set_position(z=tip_starting_z + 10, speed=spd / 3, wait=True, timeout=10)


def pipette(precursor, vol, count):
    a, b = vol // tip_max_vol, vol % tip_max_vol
    # print(a, b)
    for i in range(int(a)):
        get_liquid(precursor, tip_max_vol)
        dispense(tip_max_vol, count)
    if b != 0:
        get_liquid(precursor, b)
        dispense(b, count)


def get_liquid(precursor, vol):
    spd = 10000
    if precursor == "solute":
        x, y, z = solute_x, solute_y, solute_z
    elif precursor == "solvent":
        x, y, z = solvent_x, solvent_y, solvent_z
    else:
        raise Exception("Please enter solute or solvent", precursor)
    u1.set_position(z=max_z, speed=spd, wait=True, timeout=10)
    u1.set_position(x, y, speed=spd, wait=True, timeout=10)
    u1.set_position(z=z, speed=spd, wait=True, timeout=10)
    u1.set_3d_feeding(distance=-(vol + sip_more + sip_offset), speed=spd, relative=True, timeout=60)
    u1.set_3d_feeding(distance=sip_offset, speed=spd, relative=True, timeout=60)
    u1.set_position(z=z + 40, speed=spd, wait=True, timeout=10)


def dispense(vol, count):
    well_x, well_y = well_coords[count]
    u1.set_position(z=well_starting_z + 80, speed=spd, wait=True, timeout=10)
    u1.set_position(x=well_starting_x + well_x, y=well_starting_y + well_y, speed= spd, wait=True, timeout=10)
    u1.set_position(z=well_starting_z, speed=spd, wait=True, timeout=10)
    u1.set_3d_feeding(distance=vol, speed=spd, relative=True, timeout=60)
    u1.set_position(z=well_starting_z + 80, speed=spd, wait=True, timeout=10)
    dispose_liq(sip_more)
    # u1.set_3d_feeding(distance=-30, speed=spd, relative=True, timeout=60)


def dispose_liq(vol):
    u1.set_position(z=max_z, speed=spd, timeout=20, wait=True)
    u1.set_position(x=trash_liq_x, y=trash_liq_y, speed= spd, timeout=20, wait=True)
    u1.set_position(z=trash_liq_z + 10, speed=spd, timeout=20, wait=True)
    u1.set_3d_feeding(distance=40 + vol, speed=spd, relative=True, timeout=60)
    u1.set_3d_feeding(distance=-40, speed=spd, relative=True, timeout=60)


def mixing(count, times):
    well_x, well_y = well_coords[count]
    u1.set_position(z=max_z, speed= spd, wait=True, timeout=10)
    u1.set_position(x=well_starting_x + well_x, y=well_starting_y + well_y, speed= spd, wait=True, timeout=10)
    u1.set_position(z=well_starting_z, speed=spd, wait=True, timeout=10)
    for i in range(times):
        u1.set_3d_feeding(distance=-180, speed=spd, relative=True, timeout=60)
        u1.set_3d_feeding(distance=180, speed=spd, relative=True, timeout=60)
    u1.set_3d_feeding(distance=40, speed=spd, relative=True, timeout=60)
    u1.set_position(z=well_starting_z + 60, speed=spd, wait=True, timeout=10)
    u1.set_3d_feeding(distance=-40, speed=spd, relative=True, timeout=60)


def dispose_tip():
    u1.set_position(z=max_z, speed= spd, timeout=20, wait=True)
    u1.set_position(x=trash_x, y=trash_y, speed= spd, timeout=20, wait=True)
    u1.set_position(z=trash_z, speed=spd, timeout=20, wait=True)
    sleep(1)
    u1.set_wrist(10, wait=True)
    sleep(1)
    u1.set_wrist(170, wait=True)
    sleep(1)
    u1.set_wrist(90, wait=True)
    sleep(1)


def move_well(u, count):
    if u == 1:
        well_x, well_y = well_coords[count]
        u1.set_position(z=max_z, speed=spd, wait=False, timeout=10)
        u1.set_position(x=well_starting_x + well_x, y=well_starting_y + well_y, speed=spd, wait=False, timeout=10)
        u1.set_position(z=well_starting_z + 12, speed=spd, wait=False, timeout=10)

    elif u == 2:
        # well_y, well_x = well_coords[count]
        # # compensate the z difference along the diagonal of the well plate
        # z_offset_x, z_offset_y = (count) % 12, (count) // 12
        # dist_ratio = (z_offset_x ** 2 + z_offset_y ** 2) / (11 ** 2 + 7 ** 2)
        # z_difference = 3.5
        # z_offset = dist_ratio * z_difference
        # u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
        # u2.set_position(well_starting_x_u2 - well_x, well_starting_y_u2 + well_y, speed=spd, wait=False,
        #                 timeout=10)
        # u2.set_position(z=well_starting_z_u2 - z_offset + 15, speed=spd, wait=False, timeout=10)
        current_coords = offset(total_count, count, well_starting_u2, well_final_u2)
        u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
        u2.set_position(x=current_coords[0], y=current_coords[1], speed=spd, wait=False,
                        timeout=10)
        u2.set_position(z=current_coords[2] + 12, speed=spd, wait=False, timeout=10)


def measure(count, camera_id=0):
    # well_y, well_x = well_coords[count]
    # compensate the z difference along the diagonal of the well plate
    # z_offset_x, z_offset_y = (count) % 12, (count) // 12
    # dist_ratio = (z_offset_x ** 2 + z_offset_y ** 2) / (11 ** 2 + 7 ** 2)
    # z_difference = 3.5
    # z_offset = dist_ratio * z_difference
    current_coords = offset(total_count, count, well_starting_u2, well_final_u2)
    u2.set_position(z=max_z_u2, speed=spd, wait=True, timeout=10)
    u2.set_position(x=current_coords[0], y=current_coords[1], speed=spd, wait=True,
                    timeout=10)
    u2.set_position(z=current_coords[2] + 12, speed=spd, wait=True, timeout=10)
    sleep(1)
    # while True:
    #     u2.send_cmd_sync("M2019")
    #     sleep(1)
    #     u2.send_cmd_sync("M17")
    #     sleep(1)
    #     if not cv_digits.cleaned(camera_id):
    #         break
    u2.send_cmd_sync("M2019")
    sleep(1)
    u2.send_cmd_sync("M17")
    sleep(1)
    # u2.set_position(z=5, relative=True)
    # sleep(1)
    # u2.send_cmd_sync("M2019")
    # sleep(0.5)
    # u2.send_cmd_sync("M17")
    u2.set_position(z=1, relative=True)


def offset(total_count, current_count, starting_coords, final_coords):
    current_coords = []
    for i in range(3):
        diff = final_coords[i] - starting_coords[i]
        offset = diff / total_count * current_count
        current_coords.append(starting_coords[i] + offset)
    return current_coords


def clean():
    u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
    u2.set_position(x=ultrasonic_1[0], y=ultrasonic_1[1], speed=spd, wait=False, timeout=10)
    u2.set_position(z=ultrasonic_1[2], speed=spd, wait=True, timeout=10)
    sleep(10)
    u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
    u2.set_position(x=ultrasonic_2[0], y=ultrasonic_2[1], speed=spd, wait=False, timeout=10)
    u2.set_position(z=ultrasonic_2[2], speed=spd, wait=False, timeout=10)


def fan():
    u2.set_position(z=max_z_u2, speed=spd, wait=False, timeout=10)
    u2.set_position(x=fan_x, y=fan_y, speed=spd, wait=False, timeout=10)
    u2.set_position(z=fan_z, speed=spd, wait=False, timeout=10)


def meter_cleaned(camera_id=0):
    while not cv_digits.cleaned(camera_id):
        print('Conductivity meter is not cleaned yet...')
        sleep(10)
    print('Conductivity meter is cleaned')
