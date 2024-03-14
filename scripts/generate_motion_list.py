import random


# 73 86
motion_on_chair = [
    "端正坐",
    "翘⼆郎腿",
    "俯下⾝⼦坐",
    "伸直腿",
    "两脚跟踩底座",
    "跪坐⾯对椅背",
    "跪坐侧对椅背",
    "跪坐背对椅背",

    # 扶手
    "双⼿撑着扶⼿屁股离开底座",
    "双⼿撑着底座屁股离开底座",
    "坐在扶⼿上",
    "脚踩在底座",
    "手臂放在扶手上",

    # 椅背
    "倒挂",
    "背靠椅背",
    "侧⾝⼿扶椅背向后看",
    "侧⾝⼿扶椅背向后看",
    "⼆郎腿+⼿臂扶着椅背+⼿⽀撑头",
    "靠背，调整椅子",
    "坐在椅背上",

    # "平躺",
    # "侧躺",

    "反⽅向坐",
    "⽉下独酌",
    "屁股快滑下椅⼦",
    "坐在地上靠着椅子",
    "椅⼦底座⾼脚不着地",
    "葛优躺",
    "N-pose",
    "躺地上",
    "V-pose",
    "叹息摇头",


    "改变双腿的姿势",
    "从地上捡起物品",
    "与旁边的人交谈",
    "与你身后的人交谈",
    "伸懒腰",
    "挠头",
    "低着头思考",
    "脖子感觉不舒服",
    "背部感觉不舒服",
    "摸摸肩膀",
    "拍拍小腿",

    "玩⼿机",
    "向前挪动椅子",
    "向后挪动椅子",
    "向左挪动椅子",
    "向右挪动椅子",
    "向左扭转坐姿",
    "向右扭转坐姿",
    "向左旋转椅子",
    "向右旋转椅子",
    "向后旋转椅子",
]
# 13
motion_off_chair = [
    "围着椅⼦⾛⼀圈",
    "围着椅⼦⾛半圈，然后回来",
    "径直⾛远，然后回来",
    "去远处搬回物品",
    "离开椅子，伸展",
    "离开椅子跳⼀跳",
    "离开椅子，体前屈",
    "推倒椅⼦再扶起椅子",
    "站立并拖拽椅子到你的左边",
    "站立并拖拽椅子到你的右边",
    "站立并向前拖拽椅子",
    "站立并向后拖拽椅子",
    "站立并旋转椅子"
]


def get_random_motion_sequence():
    # 选择4-5个元素来自集合A，1-2个元素来自集合B
    a_count = 4
    b_count = 2

    selected_a = random.sample(motion_on_chair, a_count)
    selected_b = random.sample(motion_off_chair, b_count)

    selected_elements = selected_a + selected_b
    random.shuffle(selected_elements)

    return selected_elements


def write_motion_sequence():
    with open("motion_list.txt", "w", encoding='utf-8') as f:
        for i in range(60):
            motion_sequence = get_random_motion_sequence()
            for single_motion in motion_sequence:
                f.write(str(single_motion) + "--")
            f.write("\n")


write_motion_sequence()
