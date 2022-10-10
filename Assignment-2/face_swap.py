import cv2 as cv
import numpy as np
import dlib


def extract_index_nparray(nparray):
    ind = None
    for num in nparray[0]:
        ind = num
        break
    return ind


def calc_bary(pt, a, b, c):
    ab = b - a
    ac = c - a
    ap = pt - a
    nac = np.array([a[1] - c[1], c[0] - a[0]])
    nab = np.array([a[1] - b[1], b[0] - a[0]])

    beta = np.dot(ap, nac) / np.dot(ab, nac)
    gamma = np.dot(ap, nab) / np.dot(ac, nab)
    alpha = 1 - beta - gamma

    return alpha, beta, gamma


def get_affine_matrix(p1, p2, p3, p1p, p2p, p3p):
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    (x1p, y1p) = p1p
    (x2p, y2p) = p2p
    (x3p, y3p) = p3p

    m = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    m_inv = np.linalg.inv(m)
    x_p = np.array([[x1p], [x2p], [x3p]])
    y_p = np.array([[y1p], [y2p], [y3p]])
    abc = np.dot(m_inv, x_p)
    de = np.dot(m_inv, y_p)
    result = np.hstack((abc, de, np.array([[0], [0], [1]]))).T
    return result


transform = get_affine_matrix((1, 2), (2, 3), (4, 1), (5, 6), (5, 3), (2, 4))
ans = np.dot(transform, np.array([4, 1, 1]))
print(ans)

img1_c = cv.imread("Images\\Part 1\\bradley.png")
img1 = cv.cvtColor(img1_c, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(img1)
img2_c = cv.imread("Images\\Part 1\\ryan.png")
img2 = cv.cvtColor(img2_c, cv.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Arnav Singh\\PycharmProjects\\DigitalImageAnalysis"
                                 "\\shape_predictor_68_face_landmarks.dat")
faces1 = detector(img1)
for face in faces1:
    landmarks = predictor(img1, face)
    global landmark_points1, indexes_triangles
    landmark_points1 = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points1.append((x, y))

        # cv.circle(img1_c, (x, y), 1, (0, 0, 255), -1)

    points = np.array(landmark_points1, np.int32)
    convexHull = cv.convexHull(points)

    # print(convexHull)
    # cv.polylines(img1_c, [hull], True, (255, 0, 0, ), 1)
    cv.fillConvexPoly(mask, convexHull, 255)
    face_image_1 = cv.bitwise_and(img1_c, img1_c, mask)

    rect = cv.boundingRect(convexHull)

    subdiv = cv.Subdiv2D(rect)
    subdiv.insert(landmark_points1)
    triangles = subdiv.getTriangleList().astype("int32")

    # print(triangles)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    # print(indexes_triangles)
    # cv.line(img1_c, pt1, pt2, (0, 0, 255), 1)
    # cv.line(img1_c, pt2, pt3, (0, 0, 255), 1)
    # cv.line(img1_c, pt1, pt3, (0, 0, 255), 1)

faces2 = detector(img2)
for face in faces2:
    landmarks = predictor(img2, face)
    landmark_points2 = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points2.append((x, y))

        # cv.circle(img2_c, (x, y), 1, (0, 255, 255), -1)

for tr_ind in indexes_triangles:
    tr1_pt1 = landmark_points1[tr_ind[0]]
    tr1_pt2 = landmark_points1[tr_ind[1]]
    tr1_pt3 = landmark_points1[tr_ind[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv.boundingRect(triangle1)
    (x1, y1, w1, h1) = rect1
    cropped_triangle1 = img1_c[y1: y1 + h1, x1:x1 + w1]
    cropped_tr1_mask = np.zeros_like(cropped_triangle1)

    cv.line(img1_c, tr1_pt1, tr1_pt2, (0, 0, 255), 1)
    cv.line(img1_c, tr1_pt2, tr1_pt3, (0, 0, 255), 1)
    cv.line(img1_c, tr1_pt3, tr1_pt1, (0, 0, 255), 1)

    tr2_pt1 = landmark_points2[tr_ind[0]]
    tr2_pt2 = landmark_points2[tr_ind[1]]
    tr2_pt3 = landmark_points2[tr_ind[2]]

    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv.boundingRect(triangle2)
    (x2, y2, w2, h2) = rect2
    cropped_triangle2 = img2_c[y2: y2 + h2, x2:x2 + w2]

    cv.line(img2_c, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
    cv.line(img2_c, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
    cv.line(img2_c, tr2_pt3, tr2_pt1, (0, 0, 255), 1)

# cv.imshow("Bradley", img1_c)
# cv.imshow("Ryan", img2_c)
# cv.imshow("Cropped triangle 1", cropped_triangle1)
# cv.imshow("Cropped triangle 2", cropped_triangle2)
# cv.waitKey(0)
# cv.destroyAllWindows()
