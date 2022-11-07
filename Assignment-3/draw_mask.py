import cv2 as cv
import pyautogui as pag

ix, iy, sx, sy = -1, -1, -1, -1


def draw_mask(event, x, y, flags, params):
    global ix, iy, sx, sy
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 3, (0, 0, 127), -1)

        if ix != -1:
            cv.line(img, (ix, iy), (x, y), (0, 0, 127), 2)
        else:
            sx, sy = x, y
        ix, iy = x, y

    elif event == cv.EVENT_LBUTTONDBLCLK:
        ix, iy = -1, -1
        cv.line(img, (x, y), (sx, sy), (0, 0, 127), 2)


def read_image(path):
    im = cv.imread(path)
    return im


if __name__ == '__main__':
    path = 'Images\\lena256.png'
    img = read_image(path)
    # draw_line_widget = DrawLineWidget(img)
    cv.namedWindow("image")
    cv.setMouseCallback("image", draw_mask)
    while True:
        cv.imshow("image", img)
        key = cv.waitKey(1)
        # Close program with keyboard 'q'
        if key == ord("q"):
            cv.destroyAllWindows()
            break
        elif key == ord("r"):
            ix, iy, sx, sy = -1, -1, -1, -1
            img = read_image(path)
