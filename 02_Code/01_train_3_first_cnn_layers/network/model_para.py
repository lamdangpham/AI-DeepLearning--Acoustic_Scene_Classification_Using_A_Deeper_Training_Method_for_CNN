


class model_para(object):

    def __init__(self):

        #======================= Trainging parameters
        self.n_class            = 15  # Final output classes 
        self.n_output           = 15  # Final output   
        self.l2_lamda           = 0.0001  # lamda prarameter

        #========================  Input parameters
        self.n_freq             = 128  #height
        self.n_time             = 128  #width
        self.n_chan             = 1  #width
