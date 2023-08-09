from get_angle_data import GetData

data_path = ["./DoMySelf/dancy/video_data/SclassChallenge.mp4", "./DoMySelf/dancy/video_data/KnockChallenge.mp4"]
down_url = "https://www.youtube.com/watch?v=pAcShxBxNg0&t=83s"

get = GetData(save_path = "./validation_video.mp4", data_path = data_path, save_csv_path='./validation.csv', youtube_url=down_url)
get.download_youtube_video()
# get.make_angle_df()

get = GetData(data_path = data_path, save_csv_path='./train.csv')
get.make_angle_df()