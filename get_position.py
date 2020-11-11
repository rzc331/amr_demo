import sys, os
from time import sleep
from uarm.wrapper import SwiftAPI

def report_position(position):
    print("x:{x} y:{y} z:{z} ". format(x = position[0], y = position[1], z = position[2]))
u = int(sys.argv[1])
u1 = SwiftAPI(port="/dev/ttyACM0")
u2 = SwiftAPI(port="/dev/ttyACM1")

# adjust end effector offset
u1.send_cmd_sync("M2411 S94")
u2.send_cmd_sync("M2411 S74")

uarm = [u1, u2]

sleep(2)
uarm[u - 1].send_cmd_sync("M2019")
uarm[u - 1].register_report_position_callback(report_position)
uarm[u - 1].set_report_position(1)

# while(True):
#     pass
time = int(sys.argv[2])
sleep(time)
uarm[u - 1].send_cmd_sync("M17")
