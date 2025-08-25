import cv2

def play_mp4(file_path):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Video Player', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    file_path = 'output.mp4'
    play_mp4(file_path)
