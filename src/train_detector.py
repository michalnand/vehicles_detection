import AILibs

from libs.dataset_magnetometer import *
from libs.magnetometer_augmentation import *
from model                import * 
   

class RNNDetectorConfig:


    def __init__(self):

        #dataset_root_path = "/users/michal/datasets/car_detection_2/"
        #self.device       = "cpu"

        dataset_root_path = "/home/michal/datasets/car_detection_2/"
        self.device       = "cuda"

        self.results_path = "./results/rnn_model_detector/"


        self.training_dataset = self._create_train_dataset(dataset_root_path)
        self.testing_datset   = self._create_test_dataset(dataset_root_path)


        self.learning_rate = 0.001
        self.batch_size    = 128

        self.num_epoch     = 5
        self.num_steps     = (self.num_epoch*len(self.training_dataset))//self.batch_size

        self.num_testing_samples = 10000

        self.model = RnnModelDetector(3, 1, 32) 
        self.model.to(self.device)  

        self.augmentations = MagnetometerAugmentation()


        print("total training samples ", len(self.training_dataset))
        print("learning_rate          ", self.learning_rate)
        print("batch_size             ", self.batch_size)
        print("num_epoch              ", self.num_epoch)
        print("num_steps              ", self.num_steps)
        print("\n\n")
        print(self.model)
        print("\n\n")
        
        
    def _create_train_dataset(self, dataset_root_path):

        datasets = []
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Bytca/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Kysuce/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Martin_1/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Martin_2/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_05_22-Pribovce_xyz/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_01-Lietavska_Lucka/01/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_03-Kinekus/1/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_03-Kinekus/2/"))
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Porubka_03_06_2020/"))


        dataset = AILibs.DatasetCollator(datasets)

        print("num samples ", len(dataset))

        return dataset
    
    def _create_test_dataset(self, dataset_root_path):

        datasets = []
        datasets.append(DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_01-Lietavska_Lucka/02/"))

        dataset = AILibs.DatasetCollator(datasets)

        print("num samples ", len(dataset))

        return dataset



if __name__ == "__main__":

    config = RNNDetectorConfig()

    pipeline = AILibs.SegmentationTrainingPipeline(config)
    pipeline.run()