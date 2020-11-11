# USAGE
# python recognize_digits.py

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import datetime
from time import sleep
import os


def run(camera_id):
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    os.makedirs('picture_taken/{}'.format(nowTime))

    # convergence test
    while True:
        a = read(camera_id, nowTime)
        print("***",
              "{} is the first result, test convergence after 5 sec...".format(a),
              "***", sep='\n')
        sleep(5)
        b = read(camera_id, nowTime)
        if abs(a - b) <= 0.005 * a or abs(a - b) <= 5:
            print("***",
                  "{} is the second result, convergence confirmed".format(b),
                  "***", sep='\n')
            break
        else:
            print("***", a, b, "Not converged yet. Try again...", "***", sep='\n')
            # sleep(3)

    return b


def get_camera_id():
    camera_id = 0
    while True:
        try:
            cam = cv2.VideoCapture(camera_id)
            s, image = cam.read()
            img = imutils.resize(image, height=500)
        except AttributeError:
            if camera_id < 10:
                camera_id += 1
                print("Reconnecting camera with ID={}...".format(camera_id))
            else:
                print("Failed to connect camera")
                camera_id = -1
                break
        else:
            break
    return camera_id


def take_picture(should_save=False, d_id=0, nowTime=''):
  # nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  cam = cv2.VideoCapture(d_id)
  save_add = "picture_taken/{}/00_raw.jpg".format(nowTime)
  for i in range(20):
    s, img = cam.read()
    i += 1
  cam.release()
  if s:
    if should_save:
      cv2.imwrite(save_add,img)
      # cv2.imwrite("picture_taken/00_raw.jpg", img)
    print("Picture successfully taken")
  return img


def take(i):
    while True:
        take_picture(True, i)
        # sleep(2)


def read(camera_id, nowTime='0'):
    # define the dictionary of digit segments so we can identify
    # each digit on the thermostat
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
        (0, 0, 0, 0, 0, 0, 0): 0
    }

    class DecimalError(Exception):
        """Error for not detecting decimal"""
        pass

    class NoLiquidError(Exception):
        """No liquid in the sensor"""
        pass

    class OverExposedError(Exception):
        """Over exposed in binary step"""
        pass

    while True:
        try:
            image = take_picture(True, camera_id, nowTime)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)

            # load the example image
            # image = cv2.imread("test_photo/test_16.jpg")
            # pre-process the image by resizing it
            image = imutils.resize(image, height=500)
            # RGB to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # get 4 purple locating points
            # ref pure purple lower_purple = np.array([125, 43, 46])
            lower_purple = np.array([120, 43, 46])
            # ref pure purple upper_purple = np.array([155, 255, 255])
            upper_purple = np.array([155, 255, 255])
            purple = cv2.inRange(hsv, lower_purple, upper_purple)
            # blur the edge
            # purple = cv2.GaussianBlur(purple, (5, 5), 1)
            cv2.imwrite('picture_taken/{}/01_purple_points.jpg'.format(nowTime), purple)
            # cv2.imwrite('picture_taken/01_purple_points.jpg'.format(nowTime), purple)

            # morph
            hline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4), (-1, -1))
            vline = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1), (-1, -1))
            thresh_p = cv2.morphologyEx(purple, cv2.MORPH_OPEN, vline)
            thresh_p = cv2.morphologyEx(thresh_p, cv2.MORPH_OPEN, hline)
            thresh_p = cv2.morphologyEx(thresh_p, cv2.MORPH_CLOSE, hline)
            thresh_p = cv2.morphologyEx(thresh_p, cv2.MORPH_CLOSE, vline)
            cv2.imwrite('picture_taken/{}/01_purple_points_thresh.jpg'.format(nowTime), thresh_p)
            # cv2.imwrite('picture_taken/01_purple_points_thresh.jpg'.format(nowTime), thresh_p)

            # cv2.imshow("purple", purple)
            # cv2.waitKey(0)
            # find contour of the locating points
            cnts = cv2.findContours(thresh_p.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # loop over the contours
            locating_points = []
            image_copy = image.copy()
            for c in cnts:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                # if the contour is sufficiently large, it must be a digit
                if w >= 10 and (h >= 10 and h <= 30):
                    # compute the center of the contour
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    locating_points.append([cX, cY])
                    # draw the contour and center of the shape on the image
                    cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 2)
                    cv2.circle(image_copy, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(image_copy, "center", (cX - 20, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # 		#
            # 		# # # show the image
            # 		cv2.imshow("Image", image)
            # 		cv2.waitKey(0)
                    cv2.imwrite('picture_taken/{}/02_locating_points.jpg'.format(nowTime), image_copy)
                    # cv2.imwrite('picture_taken/02_locating_points.jpg'.format(nowTime), image_copy)
            # print(1)
            # change center point to appropriate matrix form
            ctr = np.array(locating_points).reshape((-1, 1, 2)).astype(np.int32)
            # transform the perspective of the image
            wrapped = four_point_transform(image, ctr.reshape(4, 2))
            # cv2.imshow("wrapped", wrapped)
            # cv2.waitKey(0)
            cv2.imwrite('picture_taken/{}/03_wrapped.jpg'.format(nowTime), wrapped)
            # cv2.imwrite('picture_taken/03_wrapped.jpg'.format(nowTime), wrapped)

            # convert to grey scale
            gray = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)

            lower_black = np.array([0, 0, 0])
            # upper_black = np.array([180, 255, 105])
            upper_black = np.array([180, 255, 35])
            black = cv2.inRange(wrapped, lower_black, upper_black)

            # cv2.imshow("black", black)
            # cv2.waitKey(0)
            cv2.imwrite('picture_taken/{}/04_black.jpg'.format(nowTime), black)

            # morph open
            hline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4), (-1, -1))
            vline = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1), (-1, -1))
            thresh = cv2.morphologyEx(black, cv2.MORPH_OPEN, vline)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hline)
            # cv2.imshow("thresh_open", thresh)
            # cv2.waitKey(0)
            # morph close
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, hline)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vline)
            # cv2.imshow("thresh_close", thresh)
            # cv2.waitKey(0)
            cv2.imwrite('picture_taken/{}/05_thresh.jpg'.format(nowTime), thresh)
            cv2.imwrite('picture_taken/display/05_thresh.jpg', thresh)

            # top left point of the purple locating point

            # print(starting_x, starting_y)
            thresh_h, thresh_w = thresh.shape
            # print(thresh_h, thresh_w)
            w = int(0.16 * thresh_w)
            h = int(0.83 * thresh_h)
            gap = int(0.235 * thresh_w)
            (starting_x, starting_y) = (int(0.048 * thresh_w), int(0.165 * thresh_h))
            digits = []
            for i in range(4):
                (x, y) = (starting_x + gap * i, starting_y)
                roi = thresh[y:y + h, x:x + w]
                # compute the width and height of each of the 7 segments
                # we are going to examine
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.18), int(roiH * 0.13))
                dHC = int(roiH * 0.05)
                # rectangle the region of each number
                rec_RGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(rec_RGB, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow("dot_position", rec_RGB)
                # cv2.waitKey(0)
                cv2.imwrite('picture_taken/{}/06_rec_' + str(i+1) + '.jpg'.format(nowTime), rec_RGB)

                # define the set of 7 segments
                segments = [
                    ((dW, 0), (w - dW, dH)),  # top
                    ((0, dH), (dW, h // 2 - dHC)),  # top-left
                    ((w - dW, dH), (w, h // 2 - dHC)),  # top-right
                    ((dW, (h // 2) - dHC), (w - dW, (h // 2) + dHC)),  # center
                    ((0, h // 2 + dHC), (dW, h - dH)),  # bottom-left
                    ((w - dW, h // 2 + dHC), (w, h - dH)),  # bottom-right
                    ((dW, h - dH), (w - dW, h))  # bottom
                ]
                # these 4 lines for test the position of segments
                RGB = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(RGB, (dW, 0), (w - dW, dH), (0, 255, 0), 2)
                cv2.rectangle(RGB, (0, dH), (dW, h // 2 - dHC), (0, 0, 255), 2)
                cv2.rectangle(RGB, (w - dW, dH), (w, h // 2 - dHC), (0, 0, 255), 2)
                cv2.rectangle(RGB, (dW, (h // 2) - dHC), (w - dW, (h // 2) + dHC), (0, 255, 0), 2)
                cv2.rectangle(RGB, (0, h // 2 + dHC), (dW, h - dH), (0, 0, 255), 2)
                cv2.rectangle(RGB, (w - dW, h // 2 + dHC), (w, h - dH), (0, 0, 255), 2)
                cv2.rectangle(RGB, (dW, h - dH), (w - dW, h), (0, 255, 0), 2)
                # cv2.imshow("position", RGB)
                # cv2.waitKey(0)
                cv2.imwrite('picture_taken/{}/07_position_'.format(nowTime) + str(i+1) + '.jpg', RGB)
                on = [0] * len(segments)

                # loop over the segments
                for (j, ((xA, yA), (xB, yB))) in enumerate(segments):
                    # extract the segment ROI, count the total number of
                    # thresholded pixels in the segment, and then compute
                    # the area of the segment
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI)
                    area = (xB - xA) * (yB - yA)

                    # if the total number of non-zero pixels is greater than
                    # 50% of the area, mark the segment as "on"
                    if total / float(area) == 1:
                        raise OverExposedError
                    elif total / float(area) > 0.3:
                        on[j] = 1

                # lookup the digit and draw it on the image
                digit = DIGITS_LOOKUP[tuple(on)]
                # if digit == -1:
                #     raise NoLiquidError
                # else:
                digits.append(digit)
                cv2.rectangle(wrapped, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(wrapped, str(digit), (x - 15, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                # cv2.imshow("Output", wrapped)
                # cv2.waitKey(0)
                cv2.imwrite('picture_taken/{}/08_digit_read.jpg'.format(nowTime), wrapped)



            # find the dot
            length = int(0.078 * thresh_h)
            # point to the top left corner of the first left dot
            # dot_x_offset = 1 + (gap + w)/2 - length/2
            # dot_y_offset = 2 + h - length
            # (dot_starting_x, dot_starting_y) = (int(top_left[0] + dot_x_offset), int(top_left[1] + dot_y_offset))
            # print(dot_starting_x, dot_starting_y)
            # dot_starting_x = int(0.24 * thresh_w)
            dot_starting_x = int(starting_x + (gap + w) / 2)
            dot_starting_y = int(starting_y + h - length)
            dot = -1
            for i in range(4):
                (x, y) = (dot_starting_x + gap * i, dot_starting_y)
                roi = thresh[y:y + length, x:x + length]
                # rectangle the region of the dot
                dot_RGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(dot_RGB, (x, y), (x + length, y + length), (0, 255, 0), 2)
                # cv2.imshow("dot_position", dot_RGB)
                # cv2.waitKey(0)
                cv2.imwrite('picture_taken/{}/09_digit_'.format(nowTime) + str(i+1) + '.jpg', dot_RGB)

                total = cv2.countNonZero(roi)
                area = length ^ 2

                # if the total number of non-zero pixels is greater than
                # 30% of the area, update the dot value
                if total / float(area) > 0.2:
                    dot = 3 - i
            # check if the decimal is detected, if not, the dot will remain as -1
            if dot == -1:
                raise DecimalError


            # display the digits
            # print(u"{}{}{}{}".format(*digits))
            number = u"{}{}{}{}".format(*digits)
            # print(number)
            if number == "0000":
                raise NoLiquidError
            final = int(number) / 10 ** dot
            # print(final)
            # cv2.imshow("Output", wrapped)
            # cv2.waitKey(0)

            # save the picture
            # nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # save_add = "picture_taken/history/" + "{}".format(str(nowTime)) + "_" +"{}".format(final) + ".jpg"
            # cv2.imwrite(save_add,image)

        except ValueError:
            print("Anchor points not appropriately detected. Try again...")
            # sleep(0.1)
        except DecimalError:
            print("Decimal not detected. Try again...")
            # sleep(0.1)
        except KeyError:
            print("Digit not recognized. Try again...")
            # sleep(0.1)
        except OverExposedError:
            print("Overexposed. Please reconnect the camera")
            final = -1
            break
        except NoLiquidError:
            print("No liquid in the conductivity sensor")
            final = 0
            break
        else:
            break
    return final


def cleaned(camera_id):
    a = read(camera_id)
    if a == 0:
        return True
    else:
        return False

