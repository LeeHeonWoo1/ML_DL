import cv2
import time
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
import yt_dlp
from typing import Union

warnings.filterwarnings("ignore")

class GetData():
    """
    유튜브 링크를 통해 데이터를 수집할 영상을 다운받아 각 전처리 과정을 거쳐 데이터셋을 구성합니다. \n
    youtube_url : 유튜브 링크 주소입니다. \n
    save_path : 영상이 저장될 위치를 의미합니다. "경로/영상을 저장할 이름.mp4" 형식으로 작성합니다. \n
    is_shorts : 해당 영상이 쇼츠 영상인지 아닌지를 불리안 값으로 입력받습니다. 기본값은 False로, shorts영상일 경우 True로 입력해야 합니다. \n
    data_path : 영상 데이터가 위치하는 경로입니다. 영상이 하나만 있을 경우에도 리스트 형태로 작성하여 입력합니다. \n
    save_csv_path : 최종 결과를 csv파일로 저장할 위치를 의미합니다.
    """
    def __init__(self, save_path : str = "", youtube_url : str = '', is_shorts : bool = False, data_path : list = None, save_csv_path : str = ""):
        self.youtube_url = youtube_url
        self.save_path = save_path
        self.is_shorts = is_shorts
        self.data_path = data_path
        self.save_csv_path = save_csv_path
        
    def download_youtube_video(self):
        """
        유튜브에서 영상을 다운받습니다. 만약 쇼츠 영상일 경우 URL을 변경합니다.
        """
        print("영상 다운로드를 시작합니다...")
        if not self.is_shorts:
            self.youtube_url = self.youtube_url.replace("shorts/", "watch?v=")
            
        ydl_opts = {
            'format': 'best',  # 최고 화질로 다운로드
            'outtmpl': self.save_path,  # 다운로드한 파일 저장 경로와 이름
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.youtube_url])
        print("영상 다운로드 완료 !")

    def get_dataframe(self):
        """
        전처리 전 단계의 데이터프레임을 생성하여, 리스트에 담아 리턴합니다.
        """
        print("영상을 캡쳐하고, 각 관절들의 3차원 좌표를 기반으로 데이터 프레임을 생성합니다.")
        # 랜드마크를 그릴 함수들 정의
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # 빈 데이터 프레임을 만들고
            df_list = []
            for i, path in enumerate(self.data_path):
                # 코드 실행시간 측정을 위한 변수1
                t1 = time.time()
                
                # 영상에서 각 정보를 추출해 영상의 길이를 측정
                cap = cv2.VideoCapture(path)
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                len_video = int(count/fps)
                
                df = pd.DataFrame()
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break
                    
                    # 코드 실행 시간 측정을 위한 변수2
                    t2 = time.time()

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    
                    if results.pose_landmarks:
                        x = []
                        for k in range(33):
                            x.append(results.pose_landmarks.landmark[k].x)
                            x.append(results.pose_landmarks.landmark[k].y)
                            x.append(results.pose_landmarks.landmark[k].z)
                            x.append(results.pose_landmarks.landmark[k].visibility)
                        
                        tmp = pd.DataFrame(x).T
                        df = pd.concat([df, tmp])
                    else:
                        continue

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

                    cv2.imshow('MediaPipe Pose', image)
                    
                    # 만약 q가 눌리거나 코드 실행 시간이 영상 시간의 길이보다 길어지면 반복문을 탈출
                    if cv2.waitKey(1) & 0xFF == ord("q") or len_video < int(t2 - t1):
                        break
                    
                df["label"] = i
                df_list.append(df)

        cap.release()
        cv2.destroyAllWindows()
        print("데이터 프레임 생성 완료 !")
        return df_list
    
    def pretreat_df(self):
        """
        1차적으로 생성한 데이터프레임을 전처리합니다.
        """
        df_list = self.get_dataframe()
        print("데이터 프레임 1차 전처리를 시작합니다.")
        
        # 각 관절을 의미하는 dict 생성
        pose_dict = {
        0 : "NOSE",             1 : "LEFT_EYE_INNER", 2 : "LEFT_EYE",    3 : "LEFT_EYE_OUTER", 4 : "RIGHT_EYE_INNER", 5 : "RIGHT_EYE",       6 : "RIGHT_EYE_OUTER",
        7 : "LEFT_EAR",         8 : "RIGHT_EAR",      9 : "MOUTH_LEFT",  10 : "MOUTH_RIGHT",   11 : "LEFT_SHOULDER",  12 : "RIGHT_SHOULDER", 13 : "LEFT_ELBOW", 14 : "RIGHT_ELBOW",
        15 : "LEFT_WRIST",      16 : "RIGHT_WRIST",   17 : "LEFT_PINKY", 18 : "RIGHT_PINKY",   19 : "LEFT_INDEX",     20 : "RIGHT_INDEX",    21 : "LEFT_THUMB", 22 : "RIGHT_THUMB",
        23 : "LEFT_HIP",        24 : "RIGHT_HIP",     25 : "LEFT_KNEE",  26 : "RIGHT_KNEE",    27 : "LEFT_ANKLE",     28 : "RIGHT_ANKLE",    29 : "LEFT_HEEL",  30 : "RIGHT_HEEL",
        31 : "LEFT_FOOT_INDEX", 32 : "RIGHT_FOOT_INDEX"
        }

        index = ["_x", "_y", "_z", "_visibility"]  # 각 관절의 좌표값을

        df_cols = []
        for i in pose_dict.values():
            for idx in index:
                df_cols.append(i+idx)              # 관절과 합쳐서
                
        temp_dict = {}
        for i in range(len(df_cols)):
            temp_dict[i] = df_cols[i]
            
        temp_list = []
        for df in df_list:
            df1 = df.rename(columns=temp_dict) # 데이터프레임의 컬럼으로 사용
            df2 = df1.set_index([pd.Index(list(range(df.shape[0])))]) # 인덱스 설정
            
            df3 = df2[['LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_SHOULDER_z', 'LEFT_SHOULDER_visibility', 'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y', 'RIGHT_SHOULDER_z', 'RIGHT_SHOULDER_visibility',
            'LEFT_ELBOW_x', 'LEFT_ELBOW_y', 'LEFT_ELBOW_z', 'LEFT_ELBOW_visibility', 'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'RIGHT_ELBOW_z', 'RIGHT_ELBOW_visibility', 'LEFT_WRIST_x',
            'LEFT_WRIST_y', 'LEFT_WRIST_z', 'LEFT_WRIST_visibility', 'RIGHT_WRIST_x', 'RIGHT_WRIST_y', 'RIGHT_WRIST_z', 'RIGHT_WRIST_visibility', 'LEFT_PINKY_x', 'LEFT_PINKY_y',
            'LEFT_PINKY_z', 'LEFT_PINKY_visibility', 'RIGHT_PINKY_x', 'RIGHT_PINKY_y', 'RIGHT_PINKY_z', 'RIGHT_PINKY_visibility', 'LEFT_INDEX_x', 'LEFT_INDEX_y', 'LEFT_INDEX_z',
            'LEFT_INDEX_visibility', 'RIGHT_INDEX_x', 'RIGHT_INDEX_y', 'RIGHT_INDEX_z', 'RIGHT_INDEX_visibility', 'LEFT_THUMB_x', 'LEFT_THUMB_y', 'LEFT_THUMB_z', 'LEFT_THUMB_visibility',
            'RIGHT_THUMB_x', 'RIGHT_THUMB_y', 'RIGHT_THUMB_z', 'RIGHT_THUMB_visibility', 'LEFT_HIP_x', 'LEFT_HIP_y', 'LEFT_HIP_z', 'LEFT_HIP_visibility', 'RIGHT_HIP_x', 'RIGHT_HIP_y',
            'RIGHT_HIP_z', 'RIGHT_HIP_visibility', 'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_z', 'LEFT_KNEE_visibility', 'RIGHT_KNEE_x', 'RIGHT_KNEE_y', 'RIGHT_KNEE_z', 'RIGHT_KNEE_visibility',
            'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z', 'LEFT_ANKLE_visibility', 'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z', 'RIGHT_ANKLE_visibility', 'LEFT_HEEL_x',
            'LEFT_HEEL_y', 'LEFT_HEEL_z', 'LEFT_HEEL_visibility', 'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_z', 'RIGHT_HEEL_visibility', 'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y',
            'LEFT_FOOT_INDEX_z', 'LEFT_FOOT_INDEX_visibility', 'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_z', 'RIGHT_FOOT_INDEX_visibility']]
            
            temp_list.append(df3)
            
        print("전처리 완료 !")
        return temp_list
    
    def calculate_angle(self, A, B, C, A1, B1, C1):
        """
        데이터프레임의 각 열(각 관절) 간 사이각을 계산합니다.
        """
        v1 = [A, B, C]
        v2 = [A1, B1, C1]

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cosine_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def make_angle_df(self):
        """
        모든 DataFrame에 대한 각도를 구한 다음, 하나의 데이터프레임으로 합쳐 csv파일로 저장합니다.
        """
        df_list = self.pretreat_df()
        print("마지막 단계를 실행합니다.")
        temp_list = []
        for idx, df in enumerate(df_list):
            columns = []
            angle_array = []
            for j in range(df.shape[0]):
                angle_sub_array = []
                for i in range(0, df.shape[1] - 4, 3):
                    # 각도 계산
                    before_col = df.iloc[j, i:i+3].values
                    next_col = df.iloc[j, i+3:i+6].values
                    A, B, C =  before_col
                    A1, B1, C1 = next_col
                    
                    angle = self.calculate_angle(A, B, C, A1, B1, C1)
                    angle_sub_array.append(angle)
                    
                    # 컬럼 이름 생성
                    before_origin = df.iloc[0, i : i + 3].index[0]
                    next_origin = df.iloc[0, i + 3 : i + 6].index[0]
                    before_angle_col_name = before_origin[:before_origin.rfind("_")]
                    next_angle_col_name= next_origin[:next_origin.rfind("_")]
                    
                    final_col_name = f"{before_angle_col_name}-{next_angle_col_name} angle"
                    if final_col_name not in columns:
                        columns.append(final_col_name)
                    
                angle_array.append(angle_sub_array)

            angle_array = np.array(angle_array)
            angle_df = pd.DataFrame(angle_array, columns=columns, index=pd.Index(list(range(df.shape[0]))))
            angle_df["label"] = idx
            temp_list.append(angle_df)
            
        result_angle_df = pd.concat(temp_list)
        result_angle_df.to_csv(self.save_csv_path, index=False, encoding="utf8", sep=",")
        print("데이터 수집 완료 !")
        return result_angle_df