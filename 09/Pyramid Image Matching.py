import cv2
import numpy as np

def build_gaussian_pyramid(image, levels):
    """
    构建高斯金字塔。
    
    :param image: 输入图像。
    :param levels: 金字塔层数。
    :return: 高斯金字塔列表。
    """
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)  # 降采样
        pyramid.append(image)
    return pyramid

def pyramid_match(template, target, pyramid_levels):
    """
    使用金字塔法进行模板匹配。
    
    :param template: 模板图像。
    :param target: 目标图像。
    :param pyramid_levels: 金字塔层数。
    :return: 模板在目标图像中的最佳匹配位置。
    """
    # 构建金字塔
    template_pyramid = build_gaussian_pyramid(template, pyramid_levels)
    target_pyramid = build_gaussian_pyramid(target, pyramid_levels)
    
    best_match = None
    best_location = None
    
    # 从金字塔顶层（低分辨率）开始匹配
    for level in range(pyramid_levels - 1, -1, -1):
        template_resized = template_pyramid[level]
        target_resized = target_pyramid[level]
        
        # 模板匹配
        result = cv2.matchTemplate(target_resized, template_resized, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)  # 获取最佳匹配位置
        
        if best_match is None or min_val < best_match:
            best_match = min_val
            best_location = (min_loc[0] * (2 ** level), min_loc[1] * (2 ** level))
        
        # 结束当前层的匹配，继续到下一层
    
    return best_location

# 主程序
if __name__ == "__main__":
    # 读取图像
    target_image = cv2.imread("images/lenna.png", cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread("images/lenna2.png", cv2.IMREAD_GRAYSCALE)

    if target_image is None or template_image is None:
        print("Error: Could not load images.")
        exit()

    # 设置金字塔层数
    levels = 3

    # 调用匹配函数
    match_location = pyramid_match(template_image, target_image, levels)

    # 在目标图像上绘制匹配框
    h, w = template_image.shape[:2]
    top_left = match_location
    bottom_right = (top_left[0] + w, top_left[1] + h)
    result_image = cv2.rectangle(target_image.copy(), top_left, bottom_right, 255, 2)

    # 显示结果
    cv2.imshow("Matching Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
