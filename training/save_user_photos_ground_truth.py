import cv2
import shutil
from pathlib import Path

# Verified exact pixel coordinates for all 12 square-cropped user photos
# Coordinates are in (x, y) on the square_i.jpg image
ground_truth = {
    1: {
        "file": "square_1.jpg",
        "size": (900, 900),
        "holes": [
            (513.5, 465.0), (498.8, 486.5), (534.5, 490.5), (522.2, 518.7),
            (576.8, 481.4), (613.8, 483.4), (577.5, 498.5), (600.8, 505.1),
            (559.1, 513.4), (551.0, 535.5)
        ]
    },
    2: {
        "file": "square_2.jpg",
        "size": (1080, 1080),
        "holes": [
            (648.0, 658.5), (659.0, 658.5),
            (581.4, 710.4), (678.8, 708.6), (731.8, 708.5),
            (629.1, 741.1), (649.0, 741.0), (616.9, 752.8),
            (576.2, 767.8), (667.7, 766.0)
        ]
    },
    3: {
        "file": "square_3.jpg",
        "size": (1200, 1200),
        "holes": [
            (776.4, 740.4), (790.3, 740.1),
            (691.8, 809.1), (815.5, 803.9), (882.1, 802.3),
            (752.6, 847.5), (777.9, 846.7), (737.1, 863.0),
            (685.1, 883.9), (801.7, 878.5)
        ]
    },
    4: {
        "file": "square_4.jpg",
        "size": (1200, 1200),
        "holes": [
            (727.1, 628.0), (699.9, 666.2), (764.4, 673.7), (741.3, 723.3),
            (841.3, 658.0), (908.1, 661.9), (841.7, 688.3), (883.4, 700.2),
            (807.9, 714.3), (792.4, 752.9)
        ]
    },
    5: {
        "file": "square_5.jpg",
        "size": (1200, 1200),
        "holes": [
            (657.0, 110.9), (711.5, 137.5), (631.5, 172.0), (543.2, 177.9),
            (734.0, 199.2), (765.5, 210.0), (625.0, 233.0), (675.3, 251.3),
            (747.5, 266.0), (686.5, 316.8)
        ]
    },
    6: {
        "file": "square_6.jpg",
        "size": (1200, 1200),
        "holes": [
            (391.3, 433.5), (443.5, 384.0), (482.1, 428.9)
        ]
    },
    7: {
        "file": "square_7.jpg",
        "size": (1200, 1200),
        "holes": [
            (654.2, 418.8), (719.0, 443.1), (730.8, 463.2), (675.8, 497.2), (757.5, 512.4),
            (763.7, 336.0), (830.6, 396.6), (750.6, 402.8), (794.0, 446.7), (812.6, 487.7)
        ]
    },
    8: {
        "file": "square_8.jpg",
        "size": (1200, 1200),
        "holes": [
            (663.4, 700.4), (731.3, 710.7), (672.0, 769.0)
        ]
    },
    9: {
        "file": "square_9.jpg",
        "size": (1200, 1200),
        "holes": [
            (628.0, 415.0), (591.0, 520.0), (511.0, 525.0), (623.0, 546.0),
            (586.0, 591.0), (548.0, 595.0), (517.0, 599.0), (632.0, 613.0),
            (560.0, 619.0), (656.0, 673.0)
        ]
    },
    10: {
        "file": "square_10.jpg",
        "size": (1200, 1200),
        "holes": [
            (789.3, 316.3), (871.8, 304.8), (811.0, 372.0)
        ]
    },
    11: {
        "file": "square_11.jpg",
        "size": (1200, 1200),
        "holes": [
            (583.9, 130.1), (405.8, 283.5), (409.7, 402.5), (990.5, 510.0),
            (574.0, 587.0), (180.8, 676.2), (725.0, 715.0), (1005.4, 845.2),
            (559.2, 918.7), (205.0, 960.0)
        ]
    },
    12: {
        "file": "square_12.jpg",
        "size": (1200, 1200),
        "holes": [
            (708.7, 284.1), (760.7, 387.0), (669.7, 392.5),
            (545.5, 595.0), (597.0, 638.0), (651.4, 684.7),
            (488.5, 773.8), (962.4, 911.4), (805.5, 985.0),
            (639.1, 987.9)
        ]
    }
}

src_folder = Path("training/user_target_photos/square_cropped")
dst_folder = Path("training/assets/user_samples")
dst_folder.mkdir(parents=True, exist_ok=True)

# Hole box radius in pixels for 1200x1200 (approx 20 px wide = 1.7% of image)
box_radius = 11.0

for num, data in ground_truth.items():
    img_name = data["file"]
    src_img = src_folder / img_name
    w, h = data["size"]
    
    dst_img = dst_folder / f"user_photo_{num}.jpg"
    dst_lbl = dst_folder / f"user_photo_{num}.txt"
    
    shutil.copy(src_img, dst_img)
    
    # Write YOLO format
    lines = []
    bw = (box_radius * 2.2) / float(w)
    bh = (box_radius * 2.2) / float(h)
    
    for hx, hy in data["holes"]:
        cx = hx / float(w)
        cy = hy / float(h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
    dst_lbl.write_text("\n".join(lines) + "\n")
    print(f"Saved user_photo_{num}: {len(data['holes'])} holes")

print("All 12 user photos ground truth saved to training/assets/user_samples/!")

