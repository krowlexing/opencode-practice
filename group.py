import cv2

grouping_distance = 100

def box_distance(a, b):
    a_x1, a_y1, a_x2, a_y2 = list(map(int, a.xyxy[0]))
    b_x1, b_y1, b_x2, b_y2 = list(map(int, b.xyxy[0]))

    a_x = (a_x1 + a_x2) / 2
    a_y = (a_y1 + a_y2) / 2
    b_x = (b_x1 + b_x2) / 2
    b_y = (b_y1 + b_y2) / 2

    return ((a_x - b_x)**2 + (a_y - b_y)**2) ** 0.5

def box_distances(a, xs):
    return list(map(lambda x: box_distance(a, x), xs))

def collect_with_flag(flags, box_list, flag):
    result = []
    for idx, box in enumerate(box_list):
        if flags[idx] == flag:
            result.append((idx, box))
    return result

def get_unseen(flags, box_list):
    return collect_with_flag(flags, box_list, "unseen")

def get_grouped_unseen(flags, box_list):
    return collect_with_flag(flags, box_list, "grouped-unseen")

def group_boxes(box_list):
    flags = ["unseen"] * len(box_list)
    groups = []
    new_group = []
    while True:
        grouped_unseen_items = get_grouped_unseen(flags, box_list)
        if len(grouped_unseen_items) > 0:
            cur_idx, box = grouped_unseen_items[0]
        else:
            # previous group boxes fully checked => no other box is close enough to get into group
            if len(new_group) != 0:
                groups.append(new_group)
                new_group = []
            
            unseen_items = get_unseen(flags, box_list)
            if len(unseen_items) == 0:
                break
            
            cur_idx, box = unseen_items[0]
        
        if flags[cur_idx] == "unseen":
            new_group.append(box)
        
        flags[cur_idx] = "grouped-seen"

        remaining_unseen = get_unseen(flags, box_list)
        if len(remaining_unseen) == 0:
            continue
        indices, unseen = list(zip(*remaining_unseen))
        distances = box_distances(box, unseen)
        for idx, distance in zip(indices, distances):
            
            if distance < grouping_distance:
                flags[idx] = "grouped-unseen"
                new_group.append(box_list[idx])

    return groups

def save_image(image, file):
    return cv2.imwrite(image, file)

def box_to_int_coords(box):
    return list(map(int, box.xyxy[0]))

def calculate_bounding_rectangle(boxes):
    coords = list(map(box_to_int_coords, boxes))

    leftmost_x = min(map(lambda x: x[0], coords))
    top_y = min(map(lambda x: x[1], coords))
    rightmost_x = max(map(lambda x: x[2], coords))
    bottom_y = max(map(lambda x: x[3], coords))

    return ((leftmost_x, top_y), (rightmost_x, bottom_y))

def detect_groups_task(image, box_list):
    groups = group_boxes(box_list)
   
    groups_ = list(filter(lambda g: len(g) > 1, groups))
    for group in groups_:
        a, b = calculate_bounding_rectangle(group)
        cv2.rectangle(image, a, b, (0, 0, 255), 1)
        cv2.rectangle(image, (a[0], b[1]), (a[0] + 15, b[1] - 20), (255, 255, 255), -1)
        output = cv2.putText(image, str(len(group)), (a[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
    
    if len(groups_) > 0:
        save_image("output_groups_detection.jpg", output)