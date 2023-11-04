import argparse
import cv2
import numpy as np
import glob
import os
import imutils
import signal
from multiprocessing import freeze_support

def parsing(parser):
    parser.add_argument('--img-dir', required=True, help="이미지 파일 폴더")
    parser.add_argument('--mode', required=True, help="이미지 편집 모드")
    parser.add_argument('--result-dir', required=True, help='저장 폴더')

class TimeOutException(Exception):
    pass

# thread
def alarm_handler(signum, frame):
    print()
    print("시간이 초과되었습니다.")
    raise TimeOutException()

def crop(img, y1, y2, x1, x2):
    croped = img[y1:y2, x1:x2]
    return croped
# 이미지 사이즈 벗어나는 경우
# 다르게 crop (done)

def crop_ratio(img, y1, y2, x1, x2):
    croped = img[int((img.shape[0] * y1)):int((img.shape[0] * y2)), int((img.shape[1] * x1)):int((img.shape[1] * x2))]
    return croped

def resize(img, x, y, t):
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(t)
    resized = cv2.resize(img, (x, y))
    return resized
# 범위 벗어남, 비율
# resize 가능한 범위

def resize_ratio(img, fx, fy):
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(5)
    resized = cv2.resize(img, (0, 0), fx=fx, fy=fy)
    return resized

def grayscale(img):
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscaled

def rotation(img, degree):
    rotate = imutils.rotate_bound(img, degree)
    return rotate
# 45, ...
# 양 옆 여백 (done)
# find

def stack(img_1, img_2, v_h):
    if v_h == 'v':
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]
        stacked = np.zeros((h1+h2, max(w1, w2), 3), np.uint8)
        stacked[:h1, :w1, :3] = img_1
        stacked[h1:h1+h2, :w2, :3] = img_2
        return stacked
    elif v_h == 'h':
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]
        stacked = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
        stacked[:h1, :w1,:3] = img_1
        stacked[:h2, w1:w1+w2,:3] = img_2
        return stacked



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsing(parser)
    args = parser.parse_args()
    mode = args.mode

    # 폴더 없으면 생성
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    img_files = glob.glob(args.img_dir + '/*.jpg')
    lst = [0] * len(img_files)
    lst = [cv2.imread(i, cv2.IMREAD_COLOR) for i in img_files]
    
    while True:
        if mode == 'crop':
            while True:
                by = input("범위 지정(1)? 비율(2)?")
                if by == '1':
                    while True:
                        y1, y2, x1, x2 = map(int, input('y1 y2 x1 x2 입력 { (y1, x1 >= 0), (y2, x2 > 0), (y1 < y2), (x1 < x2) }: ').split())
                        if (y1 >= 0) and (y2 > 0) and (x1 >= 0) and (x2 > 0) and (y1 < y2) and (x1 < x2):
                            for i in range(len(lst)):
                                image_name = img_files[i][7:-4]
                                image = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
                            # result_img = crop(image, y1, y2, x1, x2)
                            # if (x1 <= image.shape[1]) and (x2 <= image.shape[1]) and (y1 <= image.shape[0]) and (y2 <= image.shape[0]):    # 범위 충족하는 이미지들 crop 처리
                                if (max(y1, y2) <= image.shape[0]) and (max(x1, x2) <= image.shape[1]):    # 범위 충족하는 이미지들 crop 처리
                                # cv2.imwrite(args.result_dir + '/' + j[6:-4] + '_crop.jpg', result_img)
                                    lst[i] = crop(lst[i], y1, y2, x1, x2)
                                else:
                                    while ((x1 > image.shape[1]) or (x1 < 0)) or ((x2 > image.shape[1]) or (x2 <= 0)) or ((y1 > image.shape[0]) or (y1 < 0)) or ((y2 > image.shape[0]) or (y2 <= 0)):
                                        y1, y2, x1, x2 = map(int, input(f'"{image_name}" 파일의 범위를 초과하였습니다. 범위(y1 y2 x1 x2)를 [ (0,0) 초과 {image.shape[:2]} 미만]을 입력하세요.').split())
                                    result_img = crop(image, y1, y2, x1, x2)
                                    # cv2.imwrite(args.result_dir + '/' + j[6:-4] + '_crop.jpg', result_img)
                                    lst[i] = crop(lst[i], y1, y2, x1, x2)
                            file_name = img_files
                            cv2.imshow('crop', lst[0])
                            cv2.waitKey(0)
                            break
                        break
                    break
                elif by =='2':
                    while True:
                        y1, y2, x1, x2 = map(float, input('y1 y2 x1 x2 범위 입력(0 ~ 1): ').split())
                        if (y1 >= 0) and (y2 > 0) and (x1 >= 0) and (x2 > 0) and (y1 < y2) and (x1 < x2) and (y1 <= 1) and (y2 <= 1) and (x1 <= 1) and (x2 <= 1):
                            for i in range(len(lst)):
                                lst[i] = crop_ratio(lst[i], y1, y2, x1, x2)
                                # image_name = img_files[i][7:-4]
                                # image = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
                                # result_img = crop(image, y1, y2, x1, x2)
                                # if (x1 <= image.shape[1]) and (x2 <= image.shape[1]) and (y1 <= image.shape[0]) and (y2 <= image.shape[0]):    # 범위 충족하는 이미지들 crop 처리
                                # if (max(y1, y2) <= image.shape[0]) and (max(x1, x2) <= image.shape[1]):    # 범위 충족하는 이미지들 crop 처리
                                #     # cv2.imwrite(args.result_dir + '/' + j[6:-4] + '_crop.jpg', result_img)
                                #     lst[i] = crop_ratio(lst[i], y1, y2, x1, x2)
                                # else:
                                #     while ((x1 > image.shape[1]) or (x1 < 0)) or ((x2 > image.shape[1]) or (x2 <= 0)) or ((y1 > image.shape[0]) or (y1 < 0)) or ((y2 > image.shape[0]) or (y2 <= 0)):
                                #         y1, y2, x1, x2 = map(int, input(f'"{image_name}" 파일의 범위를 초과하였습니다. 범위(y1 y2 x1 x2)를 [ (0,0) 초과 {image.shape[:2]} 미만]을 입력하세요.').split())
                                #     result_img = crop_ratio(image, y1, y2, x1, x2)
                                #     # cv2.imwrite(args.result_dir + '/' + j[6:-4] + '_crop.jpg', result_img)
                                #     lst[i] = crop_ratio(lst[i], y1, y2, x1, x2)
                            file_name = img_files
                            cv2.imshow('crop_ratio', lst[0])
                            cv2.waitKey(0)
                            break
                    break

        elif mode == 'resize':
            while True:
                by = input("크기 지정(1)? 비율(2)?")
                if by == '1':
                    while True:
                        x, y = map(int, input('x, y를 입력하세요.').split())
                        if (x > 0) and (y > 0): # 0 처리
                            t = int(input('최대 시간을 입력해주세요: '))
                            try:
                                for i in range(len(lst)):
                                    lst[i] = resize(lst[i], x, y, t)
                                signal.alarm(0)
                                cv2.imshow('resize', lst[0])
                                cv2.waitKey(0)
                                break
                            except TimeOutException as e:
                                print("""10000 x 10000을 초과하는 크기는 시간이 오래 걸립니다.\n10000 x 10000 이하의 사이즈를 입력해주세요.""")
                                print()
                                cont = input("무시하고 계속하시겠습니까? y / else(크기 다시 설정): ")
                                if cont == 'y':
                                    signal.alarm(0)
                                    for i in range(len(lst)):
                                        # lst[i] = resize(lst[i], x, y)
                                        lst[i] = cv2.resize(lst[i], (x, y))
                    break
                elif by == '2':
                    while True:
                        fx, fy = map(float, input('비율을 입력하세요. 둘 다 0보다 커야함.').split())
                        if (fx > 0) and (fy > 0):
                            try:
                                for i in range(len(lst)):
                                    lst[i] = resize_ratio(lst[i], fx, fy)
                                signal.alarm(0)
                                cv2.imshow('resize_ratio', lst[0])
                                cv2.waitKey(0)
                                break
                            except TimeOutException as e:
                                print("20배를 넘어가면,,,")
                                print()
                                cont = input("무시하고 계속하시겠습니까? y / else(크기 다시 설정): ")
                                if cont == 'y':
                                    signal.alarm(0)
                                    for i in range(len(lst)):
                                        lst[i] = cv2.resize(lst[i], (0, 0), fx=fx, fy=fy)
                    break
            file_name = img_files

        elif mode == 'rotation':
            degree = int(input('각도를 입력하세요(+: 시계, -: 반시계): '))
            for i in range(len(lst)):
                lst[i] = rotation(lst[i], degree)
            file_name = img_files
            cv2.imshow('rotation', lst[0])
            cv2.waitKey(0)

        elif mode == 'grayscale':
            for i in range(len(lst)):
                lst[i] = cv2.cvtColor(lst[i], cv2.COLOR_BGR2GRAY)
            file_name = img_files
            cv2.imshow('gray', lst[0])
            cv2.waitKey(0)

        elif mode == 'stack':
            while True:
                v_h = input('v? h?')
                if (v_h == 'v') or (v_h == 'h'):
                    result_img = lst[0]
                    for i in range(1, len(lst)):
                        result_img = stack(result_img, lst[i], v_h)
                    lst = [result_img]
                    cv2.imshow('stack', result_img)
                    cv2.waitKey(0)
                    break
                else:
                    continue

        elif mode == 'merge':
            while True:
                alpha = float(input('alpha값: '))
                if (alpha >= 0) and (alpha <= 1):
                    result_img = lst[0]
                    for i in range(1, len(lst)):
                        resized_img = resize(lst[i], result_img.shape[1], result_img.shape[0])
                        result_img = cv2.addWeighted(result_img, alpha, resized_img, (1-alpha), 0)
                        print(result_img.shape)
                    lst = [result_img]
                    cv2.imshow('merge', result_img)
                    cv2.waitKey(0)
                    break
                else:
                    print('0과 1 사이의 값 입력')
                    continue
        
        q = input("계속하시겠습니까? y: 추가적인 모드 선택 / n: 저장 후 종료 / else: 저장하지 않고 종료")
        if q == 'n':
            if mode == 'stack':
                cv2.imwrite(args.result_dir + '/' + 'stacked.jpg', lst[0])
            elif mode == 'merge':
                cv2.imwrite(args.result_dir + '/' + 'merged.jpg', lst[0])
            else:
                for i in range(len(lst)):
                    cv2.imwrite(args.result_dir + '/' + file_name[i][6:-4] + '.jpg', lst[i])
            break
        elif q == 'y':
            mode = input('모드를 입력해주세요.')
            continue
        else:
            break

