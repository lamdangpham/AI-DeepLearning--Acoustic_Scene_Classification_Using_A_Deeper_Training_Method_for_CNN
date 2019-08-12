
class dnn_bl02_para(object):

    def __init__(self):

        #=======Layer 01: full connection
        self.l01_fc             = 512 # node number of first full-connected layer 
        self.l01_is_act         = True
        self.l01_act_func       = 'RELU'
        self.l01_is_drop        = True
        self.l01_drop_prob      = 0.2

        #=======Layer 02: full connection
        self.l02_fc             = 1024  # node number of first full-connected layer 
        self.l02_is_act         = True
        self.l02_act_func       = 'RELU'
        self.l02_is_drop        = True
        self.l02_drop_prob      = 0.2

        #=======Layer 03: Final layer
        self.l03_fc             = 1024   # output node number = class numbe
        self.l03_is_act         = True
        self.l03_act_func       = 'RELU'
        self.l03_is_drop        = True
        self.l03_drop_prob      = 0.2


        #=======Layer 04: Final layer
        self.l04_fc             = 15   # output node number = class numbe
        self.l04_is_act         = False
        self.l04_act_func       = 'RELU'
        self.l04_is_drop        = False
        self.l04_drop_prob      = 1

