import numpy
import torch
import scipy.signal as signal

from matplotlib import pyplot as plt


class DatasetMagnetometerBase:
    def __init__(self, root_path):
        self.root_path = root_path

        annotation_data = numpy.genfromtxt(root_path + "/anotacie_final.csv", delimiter=';')
        data            = numpy.genfromtxt(root_path + "/s0.txt", delimiter=' ', dtype=numpy.float32)

        # class label and position in stream
        class_label      = annotation_data[1:-1, 0]
        event_position   = annotation_data[1:-1, 3]

        # extract only valid
        valid_id = numpy.where(class_label >= 0)[0]


        self.class_labels    = numpy.array(class_label[valid_id], dtype=int)
        self.event_positions = numpy.array(event_position[valid_id], dtype=int)


        #load raw magnetic field data
        x = numpy.transpose(data)[6]
        y = numpy.transpose(data)[7]
        z = numpy.transpose(data)[8]

        self.x_hp = self._hp_filter(x)
        self.y_hp = self._hp_filter(y)
        self.z_hp = self._hp_filter(z)


    def _hp_filter(self, data, alpha = 0.02):

        b = [alpha]
        a = [1.0, -(1.0 - alpha)]

        # Run the filter (zi sets the initial state so it starts at data[0])
        zi = signal.lfilter_zi(b, a) * data[0]
        ema_array, _ = signal.lfilter(b, a, data, zi=zi)

        return data - ema_array
        

    def _is_valid(self, annotation_data, idx):
        if numpy.isnan(annotation_data[idx][3]) == True:
            return False
        
        return True


class DatasetMagnetometerDetection(DatasetMagnetometerBase):

    def __init__(self, root_path):
        super().__init__(root_path)  

        print("loading ", root_path, self.get_seq_length() )
       

    def get_seq_length(self):
        return len(self.x_hp) - 1024


    def get(self, idx, window_size = 512, dist_max = 200):
        w = window_size//2
        idx  = numpy.clip(idx, window_size, len(self.x_hp) - window_size - 1)

        x = self.x_hp[idx-w:idx+w]
        y = self.y_hp[idx-w:idx+w]
        z = self.z_hp[idx-w:idx+w]

        res = numpy.stack([x, y, z])
        res = numpy.transpose(res)

        energy = numpy.sqrt((res**2).mean(axis=-1))
        energy_mean = numpy.sqrt((res**2).mean())

        mask = numpy.abs(idx - self.event_positions) < window_size

        # 2. Filter the original array using the mask
        candidates = self.event_positions[mask]

        #print("get ", candidates)
        label = numpy.zeros((window_size, 1))

        for i in candidates:
            j = numpy.clip(i - idx + w , 0, window_size-1)
            if j != 0 and j < window_size-1:

                tmp = energy[j:j+window_size//4].mean()
                
                #print("energy = ", j, energy_mean, tmp)
                if tmp > 1.1*energy_mean:
                    label[j:j+dist_max, 0] = 1.0
            
        return res, label


    def __len__(self):
        return self.get_seq_length()


    def __getitem__(self, idx):
        return self.get(idx)
    







class DatasetMagnetometerClassification(DatasetMagnetometerBase):

    def __init__(self, root_path):
        super().__init__(root_path) 
        
        print("loading     ", root_path, self.get_num_events()) 
        print("num_classes ", max(numpy.unique(self.class_labels))+1)
       
    def get_num_events(self):
        return len(self.event_positions)

    '''
        return window containing centered positive detection and label
    '''
    def get(self, event_idx, window_size=512):
        offset = self.event_positions[event_idx]
        label  = self.class_labels[event_idx]

        w = window_size//2  

        offset = numpy.clip(offset, w, len(self.x_hp) - w - 1)

        x = self.x_hp[offset-w:offset+w]
        y = self.y_hp[offset-w:offset+w]
        z = self.z_hp[offset-w:offset+w]

        res = numpy.stack([x, y, z])
        res = res.T
        return res, label
    
    
    def __len__(self):
        return self.get_num_events()
    

    def __getitem__(self, idx):
        return self.get(idx)
    

    
   
if __name__ == "__main__":

    root_path = "/users/michal/datasets/car_detection_2/Meranie_20_06_01-Lietavska_Lucka/02/"

    #dataset = DatasetMagnetometerDetection(root_path)
    #x, y = dataset.get(3000, 512)

    dataset = DatasetMagnetometerClassification(root_path)
    x, y = dataset.get(300, 512)

    print(x.shape, y.shape, y)


    plt.plot(x[:, 0])
    plt.plot(x[:, 1])
    plt.plot(x[:, 2])
    #plt.plot(y[:, 0], color="red")
    plt.show()
    