import cv2

background_color = (255, 255, 255)
text_color = (0, 0, 0)
boundary_color = (0, 255, 0)

def write_text(image, text):
    h, w, c = image.shape
    cv2.rectangle(image, (0, 0), (100, 40), (255, 255, 255), -1)
    return cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

def draw_boxes(image, boxes):
    img = image
    for box in boxes:
        img = draw_box(image, box)
    return img

def draw_box(image, box, color=None):
    if color is None:
        color = boundary_color
    x, y, x2, y2 = list(map(int, box.xyxy[0]))
    return cv2.rectangle(image, (x, y), (x2, y2), color, 1)

def extract_boxed_images(image, boxes):
    return map(lambda box: extract_boxed_image(image, box), boxes)

def extract_boxed_image(image, box):
    x, y, x2, y2 = list(map(int, box.xyxy[0]))
    return image[y:y2, x:x2]

def save_image(image, file):
    return cv2.imwrite(image, file)
