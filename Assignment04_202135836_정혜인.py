import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.data



# ======================================================================
# 앞 섹션에서 정의한 함수 (BBox 그리기)
# ======================================================================

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



# ======================================================================
# B. Intersection over Union (IoU)
# ======================================================================

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h  # 우측 하단 = 좌측 상단 + 크기

def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    # 교집합 박스 좌표
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # 겹치지 않으면 음수가 되므로 0으로 처리
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)

    intersection_area = w * h
    union_area = area_a + area_b - intersection_area  # 겹친 부분이 두 번 더해지므로 한 번 뺌

    # TODO: ZeroDivisionError 주의
    if union_area <= 0:  # 두 박스 모두 넓이가 0인 경우
        return 0.0

    return intersection_area / union_area

apples1 = cv2.imread("../src/images/two_apples1.png")
apples2 = cv2.imread("../src/images/two_apples2.png")
apples3 = cv2.imread("../src/images/two_apples3.png")
apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb)
plt.subplot(1,3,2), plt.imshow(apples2_rgb)
plt.subplot(1,3,3), plt.imshow(apples3_rgb)

for ax in plt.gcf().axes:
    ax.axis("off")

def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn

apples1 = cv2.imread("../src/images/two_apples1.png")
apples2 = cv2.imread("../src/images/two_apples2.png")
apples3 = cv2.imread("../src/images/two_apples3.png")

apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

cntr_red1, cntr_grn1 = get_contour(apples1)
cntr_red2, cntr_grn2 = get_contour(apples2)
cntr_red3, cntr_grn3 = get_contour(apples3)

bbox1 = cv2.boundingRect(cntr_red1)
bbox2 = cv2.boundingRect(cntr_grn1)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples1_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples1_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples1_rgb);

iou1 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou1)

bbox1 = cv2.boundingRect(cntr_red2)
bbox2 = cv2.boundingRect(cntr_grn2)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples2_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples2_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples2_rgb);

iou2 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou2)

bbox1 = cv2.boundingRect(cntr_red3)
bbox2 = cv2.boundingRect(cntr_grn3)

x_r, y_r, w_r, h_r = bbox1
x_g, y_g, w_g, h_g = bbox2

x_r1, y_r1 = x_r, y_r
x_r2, y_r2 = x_r+w_r, y_r+h_r
x_g1, y_g1 = x_g, y_g
x_g2, y_g2 = x_g+w_g, y_g+h_g

draw_bbox(apples3_rgb, (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
draw_bbox(apples3_rgb, (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

plt.figure(figsize=(12,8))
plt.imshow(apples3_rgb);

iou3 = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
print(iou3)

plt.figure(figsize=(16,6))

plt.subplot(1,3,1), plt.imshow(apples1_rgb), plt.title(f"IoU: {iou1:.4f}")
plt.subplot(1,3,2), plt.imshow(apples2_rgb), plt.title(f"IoU: {iou2:.4f}")
plt.subplot(1,3,3), plt.imshow(apples3_rgb), plt.title(f"IoU: {iou3:.4f}")

for ax in plt.gcf().axes:
    ax.axis("off")



# ======================================================================
# C. Non-Maximum Suppression (NMS)
# ======================================================================

from random import randint, gauss

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """

    # 임계값 이상인 BBox만 남김
    pruned_boxes = [box for box, score in zip(boxes, scores) if score >= conf_threshold]
    pruned_scores = [score for score in scores if score >= conf_threshold]

    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """

    if len(boxes) == 0:
        return [], []

    # Confidence 내림차순으로 정렬된 인덱스
    order = np.argsort(scores)[::-1]

    keep_indices = []

    while len(order) > 0:
        # 남은 BBox 중 Confidence가 가장 높은 BBox 선택
        best_index = order[0]
        keep_indices.append(best_index)

        # 선택한 BBox와의 IoU가 임계값 이하인 BBox만 남기고 반복
        order = [
            index for index in order[1:]
            if calculate_iou(boxes[best_index], boxes[index]) <= iou_threshold
        ]

    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """
    OpenCV의 cv2.dnn.NMSBoxes()를 이용해 NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록
        각 박스는 기존과 동일한 (x1, y1, x2, y2) 형식

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score
    """

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");

print(len(boxes), len(scores))

cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2)

print(len(cv2_new_boxes), len(cv2_new_boxes))

hubble_cv2_nms = skimage.data.hubble_deep_field()

for i, box in enumerate(cv2_new_boxes):
    draw_bbox(
        hubble_cv2_nms,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble_cv2_nms)
plt.axis("off");

plt.figure(figsize=(16,10))

plt.subplot(1,2,1), plt.imshow(hubble), plt.title("Custom NMS Result")
plt.subplot(1,2,2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

for ax in plt.gcf().axes:
    ax.axis("off")


plt.show()
