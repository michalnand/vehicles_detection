import numpy

class MagnetometerAugmentation:


    def __init__(self):
        pass

    def __call__(self, x, y, p_aug = 0.2):

        x = numpy.array(x)

        # amplitude scaling
        if numpy.random.rand() < p_aug:
            x[:, 0]*= numpy.random.uniform(0.25, 2.0)

        if numpy.random.rand() < p_aug:
            x[:, 1]*= numpy.random.uniform(0.25, 2.0)
        
        if numpy.random.rand() < p_aug:
            x[:, 2]*= numpy.random.uniform(0.25, 2.0)

        # add noise
        if numpy.random.rand() < p_aug:
            x+= 0.1*numpy.random.randn(x.shape[0], x.shape[1])

        # random offset
        if numpy.random.rand() < p_aug:   
            x+= 0.5*numpy.random.randn(1, x.shape[1])

        # axis flip
        if numpy.random.rand() < p_aug:
            x[:, 0]*= -1

        if numpy.random.rand() < p_aug:
            x[:, 1]*= -1    
        
        if numpy.random.rand() < p_aug:
            x[:, 2]*= -1

        # sensor rotate
        if numpy.random.rand() < p_aug:
            perm = numpy.random.permutation(3)
            x = x[:, perm]  


        return numpy.array(x), numpy.array(y)
