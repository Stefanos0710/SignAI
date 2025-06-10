import cv2

import preprocessing_live_data
import inference

if __name__ == '__main__':
    preprocessing_live_data.main()
    cv2.destroyAllWindows()
    inference.main_inference("models/trained_model_v18.keras")

