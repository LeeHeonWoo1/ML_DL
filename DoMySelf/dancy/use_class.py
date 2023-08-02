from get_angle_data import GetData

data_path = ["./validation_video.mp4"]
down_url = "https://www.youtube.com/shorts/kX0UT7noimc"

get = GetData(save_path = "./validation_video.mp4", data_path = data_path, save_csv_path='./validation.csv', youtube_url=down_url)
get.download_youtube_video()
get.make_angle_df()