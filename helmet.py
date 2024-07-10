from cv_utils import extract_boxed_image, write_text, save_image, draw_box

helmet_color = [60, 0.7, 0.6] # hsv
green_helmet_color = [131, 0.3, 0.4] # hsv

def detect_helmets_task(image, box_list):
    helmets = detect_helmets(image, box_list)
    with_helmets = helmets.count(True)
    without_helmets = len(helmets) - with_helmets

    for box, person_has_helmet in zip(box_list, helmets):
        if person_has_helmet:
            draw_box(image, box, (0, 255, 0))
        else:
            draw_box(image, box, (0, 0, 255))

    
    output = write_text(image, str(with_helmets) + "/" + str(without_helmets))
    save_image("output_helmet_detection.jpg", output)

def detect_helmets(image, box_list):
    helmets = list(map(lambda b: has_helmet(image, b), box_list))
    
    return helmets

def has_helmet(image, box):
    dist = slide(upper_third(extract_boxed_image(image, box)))
    return dist < 200

def upper_third(image):
    h, w, c = image.shape
    delta = int(h * (1 / 3))
    return image[:delta, :, :]

def average_pixel(image):
    return image.mean(0).mean(0)

def hue_dist(x, y):
    return min(abs(x[0] - y[0]), x[0] + 360 - y[0])

def total_distance(x_rgb, y_hsv):
    target_saturation = y_hsv[1]
    target_value = y_hsv[2]
    x_hsv = rgb_to_hsv(x_rgb)
    dhue = hue_dist(x_hsv, y_hsv)
    saturation = x_hsv[1]

    saturation_penalty = 0

    if (saturation < target_saturation ):
        saturation_penalty = 10000

    value_penalty = 0 if x_hsv[2] > target_value else 10000
    return dhue + saturation_penalty + value_penalty

def slide(image):
    min_distance = 1000000
    pixel = None
    h, w, c = image.shape
    for dh in range(h - 10):
        for dw in range(w - 10):
            window = image[dh:dh+10, dw:dw+10, :]
            avg_pixel = average_pixel(window)
            dist = min(total_distance(avg_pixel, helmet_color), total_distance(avg_pixel, green_helmet_color))
            if dist < min_distance:
                min_distance = dist
                pixel = avg_pixel
    
    return min_distance

def rgb_to_hsv(rgb):
    r, g, b = rgb / 255
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    h = None
    s = 0 if c_max == 0 else delta / c_max
    v = c_max
    if (delta == 0):
        h = 0
    elif (c_max == r):
        h = 60 * ((g - b) / delta % 6)
    elif (c_max == g):
        h = 60 * ((b - r) / delta + 2)
    elif (c_max == b):
        h = 60 * ((r - g) / delta + 4)
    
    return [h, s, v]