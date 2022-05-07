import cv2
import numpy as np


def nv12tobgr(file_name, height, width, start_frame):
    """
    :param file_name: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param start_frame: 起始帧
    :return: None
    """
    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置
    frame_size = height * width * 3 // 2  # 一帧图像所占用的内存字节数
    num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
    fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe

    for i in range(num_frame - start_frame):
        yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode
        img = yyyy_uv.reshape((height * 3 // 2, width)).astype('uint8')  # NV12 的存储格式为：YYYY UV 分布在两个平面（其在内存中为 1 维）
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        return bgr_img
    fp.close()
    return None


def bgr2nv12(filename, save=None):
    bgr_img = cv2.imread(filename)
    yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_YV12)
    if save is not None:
        yuv.tofile(save)
        print("save in",  save)
    return yuv


def demo():
    bgr_img = nv12tobgr('nv12.yuv', 720, 1280, 0)
    cv2.imwrite("nv12.jpg", bgr_img)
    cv2.imshow("nv12tobgr", bgr_img)
    nv12 = bgr2nv12("nv12.jpg")
    bgr_img = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    cv2.imshow("bgr2nv12", bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # bgr2nv12("2088.jpg", "2088.yuv")
    # bgr2nv12("change_line_1.jpg", "change_line_1.yuv")

    bgr_img = nv12tobgr('frame200_frame.yuv', 720, 1280, 0)
    cv2.imwrite("frame200_frame.jpg", bgr_img)
    # cv2.imshow("nv12tobgr", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

