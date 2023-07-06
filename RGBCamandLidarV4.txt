import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime
import pandas as pd
import re




# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 320,240, rs.format.z16, 30)
config.enable_stream(rs.stream.color,640,480, rs.format.bgr8, 30) ## 640,480,


# Start streaming
profile = pipeline.start(config)


# 初始化 MediaPipe Hands 和 DrawingUtils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,        min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 读取视频
cap = cv2.VideoCapture(0)

hand_result_df = pd.DataFrame(columns=['hand_idx', 'handness','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])
Handslice = pd.DataFrame(
                    columns=['hand_idx', 'handness','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])
pose_result_df = pd.DataFrame(columns=['Person ID', 'landmarkvisibility','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])
Poseslice = pd.DataFrame(columns=['Person ID', 'landmarkvisibility','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])

LabelName = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
             'MIDDLE_FINGER_TIP','RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_LFINGER_TIP','PINKY_MCP',
             'PINKY_PIP','PINKY_DIP','PINKY_TIP']
LabelName2= LabelName + LabelName
ID = list(range(0,21))
ID2 = list(range(0,21)) +list(range(0,21))
DataOutput = pd.DataFrame()
DataMatrix = pd.DataFrame()

pose_result_df_RGB = pd.DataFrame(columns=['Person ID', 'landmarkvisibility','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])
Poseslice_RGB = pd.DataFrame(columns=['Person ID', 'landmarkvisibility','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])


try:
    while True:

        # Timestamp 获取当前时间
        current_time = datetime.now()
        # 将当前时间格式化为字符串，包含毫秒
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")

        formatted_time_str = current_time.strftime("%H_%M_%S_%f")

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue


        color_image = np.asanyarray(color_frame.get_data())


        ######## eveluate based on the rgb camera  information  #####
        # 将图像输入到 MediaPipe Hands
        results = hands.process(color_image)
        # 使用 MediaPipe Pose 处理图像并获取结果
        results_pose = pose.process(color_image)
        dataArray = []

        ###########################hand detection with mediapipe and normal RGB ############################################################################

        if results.multi_hand_world_landmarks:

            Time = current_time
            h, w, c = color_image.shape
            dataSlice = str(results.multi_hand_world_landmarks)
            split_by_multiple = re.split('landmark', dataSlice)  ### |x:|y:|z:

            for id, value in enumerate(split_by_multiple):
                ValueCOR = re.findall(r': .........', value)
                for id, value in enumerate(ValueCOR):
                    if 'e' in value:
                        Datastr = ValueCOR[id]  # process later
                        Datastr = Datastr[2:-3]
                        dataArray.append(round(float(Datastr), 5) ** -5)

                        # t= print(value)
                        # print("The letter 'e' is present in the text.")
                    else:
                        Datastr = ValueCOR[id]
                        Datastr = Datastr[2:]
                        dataArray.append(round(float(Datastr), 5))

            xCor = dataArray[0::3]
            yCor = dataArray[1::3]
            zCor = dataArray[2::3]

            # tt = print(results.multi_handedness)  ###estimated probability of the predicted handedness------not accurate at all
            num_hands = len(results.multi_handedness)
            if num_hands == 1:  ### if only one hand show
                # handedness0 = str(results.multi_handedness[0].classification)
                # handedness0 = str(re.findall(r'label: ..', handedness0))
                # handedness0 = list(handedness0[-3])
                handedness0 = str(results.multi_handedness[0].classification)
                handedness0 = str(re.findall(r'index: .', handedness0))
                handedness0 = re.findall(r"\d", handedness0)
                handedness0 = float(handedness0[0])

                if handedness0 == 0:
                    hand0 = "Left"  ###estimated probability of the predicted handedness------not accurate at all
                else:
                    hand0 = "Right"  ###estimated probability of the predicted handedness------not accurate at all
                HandStr = [hand0] * 21
                Relativetimestamp = np.nan
                d = {'XCor(m)': xCor, 'YCor(m)': yCor, 'ZCor(m)': zCor, 'ID': list(range(0, 21)),
                     'LabelName': LabelName, 'Handedness': HandStr, 'TimeStamp': np.full(21, Time),
                     'RelativeTimestamp': np.full(21, Relativetimestamp)}

                try:
                    DataMatrix = pd.DataFrame(data=d)
                except:
                    t = 1
                    # throw this frame


            elif num_hands == 2:  ### if two hands show
                handedness0 = str(results.multi_handedness[0].classification)
                handedness0 = str(re.findall(r'index: .', handedness0))
                handedness0 = re.findall(r"\d", handedness0)
                handedness0 = float(handedness0[0])
                handedness1 = str(results.multi_handedness[1].classification)
                handedness1 = str(re.findall(r'index: .', handedness1))
                handedness1 = re.findall(r"\d", handedness1)
                handedness1 = float(handedness1[0])

                if int(handedness0) != int(handedness1):  ### if both left (0) and right(1) hand showed
                    if int(handedness0) == 0:
                        hand0 = "Left"  ###estimated probability of the predicted handedness------not accurate at all
                        hand1 = "Right"
                        HandStr = [hand0] * 21 + [hand1] * 21
                    else:
                        hand0 = "Right"  ###estimated probability of the predicted handedness------not accurate at all
                        hand1 = "Left"
                        HandStr = [hand0] * 21 + [hand1] * 21

                else:
                    continue  #### do nothing

                Relativetimestamp = np.nan
                d = {'XCor(m)': xCor, 'YCor(m)': yCor, 'ZCor(m)': zCor, 'ID': ID2,
                     'LabelName': LabelName2, 'Handedness': HandStr, 'TimeStamp': np.full(42, Time),
                     'RelativeTimestamp': np.full(42, Relativetimestamp)}
                try:
                    DataMatrix = pd.DataFrame(data=d)
                except:
                    #### some times the output is not always 21
                    t = 1
                    # throw this frame

            DataOutput = pd.concat([DataOutput, DataMatrix], ignore_index=True)

            currentDateAndTime = datetime.now()
            currentTimeHMS = currentDateAndTime.strftime("%H_%M")

            # print("Time stamp is", currentTimeHMS)
            DataOutput.to_csv('hand_result_mp_RGBonly' +  '.csv') # formatted_time_str +
        ###########################hand detection with mediapipe and normal RGB ############################################################################

        ###########################pose detection with mediapipe and normal RGB ############################################################################

        # cv2.imshow('RGB', color_image)
        if results_pose.pose_world_landmarks :
            # t=print(results_pose)
            # y = print(results_pose.pose_world_landmarks)
            for landmark_idx, lm in enumerate(results_pose.pose_world_landmarks.landmark):
                x= lm.x
                visibility=lm.visibility
                PersonID = 0
                Poseslice_RGB.loc[0] = [PersonID, lm.visibility, landmark_idx, formatted_time, lm.x,
                                    lm.y, lm.z]
                pose_result_df_RGB = pd.concat([pose_result_df_RGB, Poseslice_RGB])
            # print("Time stamp is", currentTimeHMS)
            pose_result_df_RGB.to_csv('pose_result_mp_RGBonly' + '.csv') # formatted_time_str +
        ###########################pose detection with mediapipe and normal RGB ############################################################################


        ######## eveluate based on the depth information######################

        # 获取深度传感器内参
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.035), cv2.COLORMAP_JET)
        cv2.imshow('depth_colormap', depth_colormap)



        colormap_resized = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))

        # Stack both images horizontally
        images = np.hstack((colormap_resized, depth_colormap))


        # 将图像输入到 MediaPipe Hands
        results = hands.process(depth_colormap)
        # 使用 MediaPipe Pose 处理图像并获取结果
        results_pose = pose.process(depth_colormap)
        ###########################hand detection with mediapipe and depth coverted RGB ############################################################################

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 遍历每个关键点
                Handslice = pd.DataFrame(
                    columns=['hand_idx', 'handness','landmark_idx', 'formatted_time', 'x_3d', 'y_3d', 'z_3d'])

                for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                    # 打印当前循环的手部关键点索引和关键点坐标
                    # print(f"Hand {hand_idx}, Landmark {landmark_idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

                    # 获取像素坐标
                    ## in rare case
                    if landmark.x >1:
                        landmark.x =1
                    elif landmark.x <0:
                        landmark.x =0

                    if landmark.y > 1:
                        landmark.y = 0.9999
                    elif landmark.y < 0:
                        landmark.y = 0.0001

                    x_pixel = int(landmark.x * depth_colormap.shape[1])
                    y_pixel = int(landmark.y * depth_colormap.shape[0])
                    # z = landmark.z  # 如果需要深度信息

                    z_depth_pixel =  depth_frame.get_distance(x_pixel, y_pixel)
                    # 将像素坐标转换为三维坐标
                    x_3d, y_3d, z_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel], z_depth_pixel)
                    Handslice.loc[0] = [hand_idx,results.multi_handedness[hand_idx], landmark_idx, formatted_time, x_3d, y_3d, z_3d]
                    hand_result_df = pd.concat([hand_result_df, Handslice])
                    A=1
        # 将DataFrame保存为CSV文件
        hand_result_df.to_csv('hand_result_mp_depthcolor'+ '.csv', index=False) # formatted_time_str +
        ###########################hand detection with mediapipe and depth coverted RGB ############################################################################


        # cv2.imshow('depth_colormap', depth_colormap)



        ###########################pose detection with mediapipe and depth coverted RGB ############################################################################

        # 检查是否检测到了姿势关键点
        if results_pose.pose_landmarks:
            # 遍历每个关键点   only works for one person now
            for landmark_id, landmark in enumerate(results_pose.pose_landmarks.landmark):
                # 获取关键点的像素坐标
                ## in rare case
                if landmark.x > 1:
                    landmark.x = 1
                elif landmark.x < 0:
                    landmark.x = 0

                if landmark.y > 1:
                    landmark.y = 0.9999
                elif landmark.y < 0:
                    landmark.y = 0.0001

                x_pixel = int(landmark.x * depth_colormap.shape[1])
                y_pixel = int(landmark.y * depth_colormap.shape[0])
                # z = landmark.z

                z_depth_pixel = depth_frame.get_distance(x_pixel, y_pixel)


                # 将像素坐标转换为三维坐标
                x_3d, y_3d, z_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel],
                                                                   z_depth_pixel)
                person_id =0
                Poseslice.loc[0] = [person_id,landmark.visibility,landmark_id,formatted_time,x_3d,y_3d,z_3d]
                pose_result_df = pd.concat([pose_result_df, Poseslice])
            pose_result_df.to_csv('pose_result_mp_depthcolor'+  '.csv', index=False)  # formatted_time_str +

            ###########################pose detection with mediapipe and depth coverted RGB ############################################################################


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 退出 while 循环后关闭所有窗口
    cv2.destroyAllWindows()


except Exception as e:
    print(f"An error occurred: {e}")
cv2.destroyAllWindows()
