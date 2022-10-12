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
    a_apb = np.cross(ap, ab)
    a_apc = np.cross(ap, ac)
    a_abc = np.cross(ab, ac)

    # nac = np.array([a[1] - c[1], c[0] - a[0]])
    # nab = np.array([a[1] - b[1], b[0] - a[0]])
    if a_abc > 0:
        beta = a_apc / a_abc
        gamma = a_apb / a_abc
        alpha = 1 - beta - gamma
        print(a_abc)
        return alpha, beta, gamma
    else:
        return None, None, None


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
    result = np.hstack((abc, de)).T
    return result


def inverse_affine(affine_matrix):
    a = np.linalg.inv(affine_matrix[0: 2, 0: 2])
    tr = -1 * affine_matrix[0: 2, 2:]
    a = np.vstack((a, np.array([0, 0])))
    tr = np.vstack((tr, np.array([1])))
    result = np.hstack((a, tr))
    return result


def warp_bary(src, tr1, tr2, out_shape):
    tr1_a = tr1[0]
    tr2_a = tr2[0]
    tr1_b = tr1[1]
    tr2_b = tr2[1]
    tr1_c = tr1[2]
    tr2_c = tr2[2]
    # print(tr1_a, tr1_b, tr1_c, tr2_a, tr2_b, tr2_c)
    result = np.zeros(out_shape, np.uint8)
    # print(src.shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            alpha, beta, gamma = calc_bary(np.array([i, j]), tr2_a, tr2_b, tr2_c)
            # print(i, j, alpha, beta, gamma)
            if alpha and beta and gamma and alpha >= 0 and beta >= 0 and gamma >= 0:
                x_p, y_p = alpha * tr1_a + beta * tr1_b + gamma * tr1_c
                # print(x_p, y_p)
                x_p = round(x_p)
                y_p = round(y_p)
                # print(x_p, y_p)
                x_p = min(x_p, src.shape[0] - 1)
                x_p = max(x_p, 0)
                y_p = min(y_p, src.shape[1] - 1)
                y_p = max(y_p, 0)
                # print(src[x_p, y_p, :])
                result[i, j, :] = src[x_p, y_p, :]
            # print(result[i, j, :])
    # cv.imshow("result", src)
    # cv.waitKey(0)
    return result

def affine_warp(src, affine_matrix, out_shape):
    result = np.zeros(out_shape)
    inv = inverse_affine(affine_matrix)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            new_x, new_y, _ = np.dot(inv, np.array([i, j, 1])).astype("int32")
            # new_x = new_x[0]
            # new_y = new_y[0]
            new_x = min(new_x, src.shape[0] - 1)
            new_x = max(new_x, 0)
            new_y = min(new_y, src.shape[1] - 1)
            new_y = max(new_y, 0)
            result[i, j, :] = src[new_x, new_y, :]
    # print(result)
    return result.astype("uint8")


# transform = get_affine_matrix((1, 2), (2, 3), (4, 1), (5, 6), (5, 3), (2, 4))
# print(transform)
# ans = np.dot(transform, np.array([[4], [1], [1]]))
# print(ans)

img1_c = cv.imread("Images\\Part 1\\ryan.png")
img1 = cv.cvtColor(img1_c, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(img1)
img2_c = cv.imread("Images\\Part 1\\george.png")
img2 = cv.cvtColor(img2_c, cv.COLOR_BGR2GRAY)

img2_new_face = np.zeros_like(img2_c)

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
    print("rect = ", rect)

    subdiv = cv.Subdiv2D(rect)
    subdiv.insert(landmark_points1)
    print(landmark_points1)
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


# def get_landmark_points(img):
#     faces = detector(img)
#     landmark_points = []
#     for face in faces:
#         landmarks = predictor(img2, face)
#         for n in range(0, 68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
#             landmark_points.append((x, y))
#
#     return landmark_points
#
#
# lp2 = get_landmark_points(img2)

for tr_ind in indexes_triangles:
    tr1_pt1 = landmark_points1[tr_ind[0]]
    tr1_pt2 = landmark_points1[tr_ind[1]]
    tr1_pt3 = landmark_points1[tr_ind[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv.boundingRect(triangle1)
    (x1, y1, w1, h1) = rect1
    cropped_triangle1 = img1_c[y1: y1 + h1, x1:x1 + w1]
    cropped_tr1_mask = np.zeros((h1, w1), np.uint8)
    points1 = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1],
                        [tr1_pt2[0] - x1, tr1_pt2[1] - y1],
                        [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)
    #
    cv.fillConvexPoly(cropped_tr1_mask, points1, 255)
    cropped_triangle1 = cv.bitwise_and(cropped_triangle1, cropped_triangle1, mask=cropped_tr1_mask)

    # cv.line(img1_c, tr1_pt1, tr1_pt2, (0, 0, 255), 1)
    # cv.line(img1_c, tr1_pt2, tr1_pt3, (0, 0, 255), 1)
    # cv.line(img1_c, tr1_pt3, tr1_pt1, (0, 0, 255), 1)

    tr2_pt1 = landmark_points2[tr_ind[0]]
    tr2_pt2 = landmark_points2[tr_ind[1]]
    tr2_pt3 = landmark_points2[tr_ind[2]]

    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv.boundingRect(triangle2)
    (x2, y2, w2, h2) = rect2
    cropped_triangle2 = img2_c[y2: y2 + h2, x2:x2 + w2]
    cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
    points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                        [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                        [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)

    cv.fillConvexPoly(cropped_tr2_mask, points2, 255)
    cropped_triangle2 = cv.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

    # cv.line(img2_c, tr2_pt1, tr2_pt2, (0, 0, 255), 1)
    # cv.line(img2_c, tr2_pt2, tr2_pt3, (0, 0, 255), 1)
    # cv.line(img2_c, tr2_pt3, tr2_pt1, (0, 0, 255), 1)

    points1 = np.float32(points1)
    points2 = np.float32(points2)

    affine_matrix = get_affine_matrix(points1[0], points1[1], points1[2], points2[0], points2[1], points2[2])
    # print(points1, points2)
    # warped = warp_bary(cropped_triangle1, points1, points2, (h2, w2, 3))
    # print(affine_matrix)
    warped = cv.warpAffine(cropped_triangle1, affine_matrix, (w2, h2))
    # break

    triangle_area = img2_new_face[y2: y2 + h2, x2:x2 + w2]
    triangle_area = cv.add(triangle_area, warped)
    img2_new_face[y2: y2 + h2, x2:x2 + w2] = triangle_area

img2_new_face_gray = cv.cvtColor(img2_new_face, cv.COLOR_BGR2GRAY)
_, background = cv.threshold(img2_new_face_gray, 1, 255, cv.THRESH_BINARY_INV)
background = cv.bitwise_and(img2_c, img2_c, mask=background)
res = cv.add(background, img2_new_face)


cv.imshow("Bradley", img1_c)
cv.imshow("Ryan", img2_c)
cv.imshow("New face", res)
# cv.imshow("Cropped triangle 1", cropped_triangle1)
# cv.imshow("Cropped triangle 2", cropped_triangle2)
# cv.imshow("warped", warped)
cv.waitKey(0)
cv.destroyAllWindows()
