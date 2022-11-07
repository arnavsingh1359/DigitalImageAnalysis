import cv2 as cv
import numpy as np
import dlib
from make_video import create_video
import os
import shutil


def inverse_affine(affine_matrix):
    a = np.linalg.inv(affine_matrix[0: 2, 0: 2])
    tr = -1 * affine_matrix[0: 2, 2:]
    a = np.vstack((a, np.array([0, 0])))
    tr = np.vstack((tr, np.array([1])))
    result = np.hstack((a, tr))
    return result


def get_affine_matrix(src_tr, dst_tr):
    (x1, y1) = src_tr[0]
    (x2, y2) = src_tr[1]
    (x3, y3) = src_tr[2]
    (x1p, y1p) = dst_tr[0]
    (x2p, y2p) = dst_tr[1]
    (x3p, y3p) = dst_tr[2]

    m = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    if np.linalg.det(m) != 0:
        m_inv = np.linalg.inv(m)
        x_p = np.array([[x1p], [x2p], [x3p]])
        y_p = np.array([[y1p], [y2p], [y3p]])
        abc = np.dot(m_inv, x_p)
        de = np.dot(m_inv, y_p)
        result = np.hstack((abc, de)).T
        return result
    else:
        return inverse_affine(get_affine_matrix(dst_tr, src_tr))


def affine_transform(source, source_tr, destination_tr, size):
    affine_matrix = cv.getAffineTransform(np.float32(source_tr), np.float32(destination_tr))
    des = cv.warpAffine(source, affine_matrix, size, None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    return des


def morph_triangle(img1, img2, img, tr1, tr2, tr, alpha):
    (x1, y1, w1, h1) = cv.boundingRect(np.float32([tr1]))
    (x2, y2, w2, h2) = cv.boundingRect(np.float32([tr2]))
    (x, y, w, h) = cv.boundingRect(np.float32([tr]))

    tr1_rect = []
    tr2_rect = []
    tr_rect = []

    for f in range(0, 3):
        tr_rect.append(((tr[f][0] - x), (tr[f][1] - y)))
        tr1_rect.append(((tr1[f][0] - x1), (tr1[f][1] - y1)))
        tr2_rect.append(((tr2[f][0] - x2), (tr2[f][1] - y2)))

    roi = np.zeros((h, w, 3), np.float32)
    cv.fillConvexPoly(roi, np.int32(tr_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[y1:y1 + h1, x1:x1 + w1]
    img2_rect = img2[y2:y2 + h2, x2:x2 + w2]
    warp_img1 = affine_transform(img1_rect, tr1_rect, tr_rect, (w, h))
    warp_img2 = affine_transform(img2_rect, tr2_rect, tr_rect, (w, h))

    img_rect = (1 - alpha) * warp_img1 + alpha * warp_img2
    r = img[y:y + h, x:x + w]

    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - roi) + img_rect * roi


def get_corners(img):
    corners = [(0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, img.shape[0] - 1), (img.shape[1] - 1, 0)]
    return corners


def extract_index_nparray(nparray):
    ind = None
    for num in nparray[0]:
        ind = num
        break
    return ind


if __name__ == "__main__":
    filename1 = "Images\\Part 1\\ryan.png"
    filename2 = "Images\\Part 1\\george.png"

    img1 = cv.imread(filename1)
    img2 = cv.imread(filename2)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:\\Users\\Arnav Singh\\PycharmProjects\\DigitalImageAnalysis"
                                     "\\shape_predictor_68_face_landmarks.dat")
    try:
        os.mkdir("Outs")
    except OSError as error:
        shutil.rmtree("Outs")
        os.mkdir("Outs")
    faces1 = detector(img1)
    landmark_points1 = []
    for face in faces1:
        landmarks = predictor(img1, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points1.append((x, y))
    corners1 = get_corners(img1)
    landmark_points1.extend(corners1)
    # print(landmark_points1)

    faces2 = detector(img2)
    landmark_points2 = []
    for face in faces2:
        landmarks = predictor(img2, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points2.append((x, y))
    corners2 = get_corners(img2)
    landmark_points2.extend(corners2)
    # print(landmark_points2)

    points1 = np.array(landmark_points1, np.int32)
    points2 = np.array(landmark_points2, np.int32)

    rect1 = cv.boundingRect(points1)
    subdiv1 = cv.Subdiv2D(rect1)
    subdiv1.insert(landmark_points1)
    triangles1 = subdiv1.getTriangleList().astype("int32")

    indexes_triangles = []
    for t in triangles1:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points1 == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points1 == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points1 == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        # cv.line(img1, pt1, pt2, (0, 0, 255), 1)
        # cv.line(img1, pt2, pt3, (0, 0, 255), 1)
        # cv.line(img1, pt1, pt3, (0, 0, 255), 1)
    # print(indexes_triangles)

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    alphas = np.linspace(0, 1, 101)

    for alpha in alphas:
        landmark_points = []
        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            landmark_points.append((x, y))

        img_morph = np.zeros(img1.shape, dtype=img1.dtype)

        for tr_ind in indexes_triangles:
            x, y, z = tr_ind
            # print(x, y, z)
            tr1 = [landmark_points1[x], landmark_points1[y], landmark_points1[z]]
            tr2 = [landmark_points2[x], landmark_points2[y], landmark_points2[z]]
            tr = [landmark_points[x], landmark_points[y], landmark_points[z]]

            morph_triangle(img1, img2, img_morph, tr1, tr2, tr, alpha)

        img_morph = img_morph.astype("uint8")
        s = str(int(100 * alpha))
        s = s.zfill(3)
        # print(s)
        out_file = "Outs\\o" + s + ".png"
        cv.imwrite(out_file, img_morph)

    create_video("Outs", "out.mp4")
    # cv.imshow("morphed", img_morph)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # break
    # cv.imshow("Source", img1)
