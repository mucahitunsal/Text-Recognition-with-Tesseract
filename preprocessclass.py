import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self):
        pass

    def rescale(self, image, scale_percent=0.8):
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def gray_filter(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def straighten_image(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if theta > np.pi / 180 * 80 and theta < np.pi / 180 * 100:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    break

        return image

    def thresholding(self, image, threshold_value=127):
        _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return thresh

    def remove_noise(self, image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def thin_font(self, image):
        kernel = np.ones((1, 1), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def thick_font(self, image):
        kernel = np.ones((1, 1), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def remove_borders(self, image, border_size=10):
        return image[border_size:-border_size, border_size:-border_size]
