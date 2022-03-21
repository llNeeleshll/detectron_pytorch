from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import torch

import cv2
import os

class DroneDetect:

    def __init__(self, experiment_output_directory, base_model, training=False):

        self.configurations = get_cfg()

        self.experiment_output_directory = experiment_output_directory

        self.configurations.merge_from_file(model_zoo.get_config_file(base_model))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not training:

            self.configurations.MODEL.WEIGHTS = os.path.join(self.experiment_output_directory, "model_final.pth")
            self.configurations.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # the testing threshold for this model
            self.configurations.MODEL.ROI_HEADS.NUM_CLASSES = 1

            self.predictor = DefaultPredictor(self.configurations)

        pass

    def train(self, train_dataset_name, train_annotation_location, train_images_location, num_of_classes, base_model, test_dataset_name = "", test_annotation_location = "", test_images_location="", has_train=False):

        register_coco_instances(name = train_dataset_name, metadata = {}, json_file = train_annotation_location, image_root = train_images_location)

        # sample_metadata = MetadataCatalog.get(train_dataset_name)
        # dataset_dicts = DatasetCatalog.get(train_dataset_name)

        ## Create configurations

        self.configurations.DATASETS.TRAIN = (train_dataset_name,)

        if has_train:
            self.configurations.DATASETS.TEST = ()   # no metrics implemented for this dataset

        self.configurations.DATALOADER.NUM_WORKERS = 2
        self.configurations.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)
        self.configurations.SOLVER.IMS_PER_BATCH = 2
        self.configurations.SOLVER.BASE_LR = 0.02
        self.configurations.SOLVER.MAX_ITER = 600   # 300 iterations seems good enough, but you can certainly train longer
        self.configurations.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
        self.configurations.MODEL.ROI_HEADS.NUM_CLASSES = num_of_classes
        self.configurations.MODEL.DEVICE = self.device

        os.makedirs(self.experiment_output_directory, exist_ok=True)

        trainer = DefaultTrainer(self.configurations)
        trainer.resume_or_load(resume=True)
        trainer.train()
        pass


    def detect(self, image, show=False, stop=True):

        if type(image) == str:
            im = cv2.imread(image) 
        else:
            im = image    

        h = im.shape[0]
        w = im.shape[1]

        rhw = h/w

        outputs = self.predictor(im)

        if show:
            v = Visualizer(im,
                            metadata={}, 
                            scale=0.8, 
                            instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = v.get_image()

            new_w = int(700/rhw)

            result_image = cv2.resize(result_image, (new_w, 700))

            cv2.imshow('detection', result_image)

            if stop:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            return outputs

    def detect_in_video(self, video_file):

        cap = cv2.VideoCapture(video_file)

        if (cap.isOpened()== False):
            return

        # Read until video is completed
        while(cap.isOpened()):

        # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:

                # Display the resulting frame
                ##cv2.imshow('Frame',frame)

                self.detect(frame, True, stop=False)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dd = DroneDetect(r'D:\Git_Repositories\detectron_pytorch\drone_detection\output\\', "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    #dd.detect('D:/Dataset/drone_images/drone_images_test/4.jpg', True)
    dd.detect_in_video('drone_images/drone_2.mp4')
