class Spiking_ResNet18_preact(nn.Module):
    def __init__(
        self,
        num_steps: int = 5,
        init_tau: float = 2.0,              # membrane decaying time constant
        init_spk_trace_tau: float = 0.5,    # spike trace decaying time constant
        init_acc_tau: float = 2.0,          # accumulative membrane decaying time constant
        init_parametric_tau = True,
        init_version = 'v1',
        init_decay_acc = False,
        scale = 64, 
        subthresh  = 0.5,
        init_spk_trace_th = -0.35,
        init_spk_trace_a = -0.8
    ):
        super().__init__()
        self.num_steps = num_steps
        self.scale = scale
        self.subthresh = subthresh

        self.conv1 = nn.Conv2d(3, self.scale, 3, stride=1, padding=1, bias=False)   
        #
        
        self.SRB1_bn1 = nn.BatchNorm2d(self.scale)
        self.pPLIF1 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_conv1 = nn.Conv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB1_bn2 = nn.BatchNorm2d(self.scale)
        self.pPLIF2 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_conv2 = nn.Conv2d(self.scale, self.scale*4, 3, stride=1, padding=1, bias=False)
        # self.SRB1_bn3 = nn.BatchNorm2d(self.scale)
        # self.pPLIF3 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB1_conv3 = nn.Conv2d(self.scale, self.scale*4, 1, stride=1, padding=0, bias=False)
        self.SRB1_skip = nn.Conv2d(self.scale, self.scale*4, 1, stride=1, bias=False)
        self.SRB1_feedback = nn.Conv2d(self.scale, self.scale, 1, stride=1, bias=False)

        self.SRB2_bn1 = nn.BatchNorm2d(self.scale*4)
        self.pPLIF3 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_conv1 = nn.Conv2d(self.scale*4, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB2_bn2 = nn.BatchNorm2d(self.scale)
        self.pPLIF4 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_conv2 = nn.Conv2d(self.scale, self.scale*4, 3, stride=1, padding=1, bias=False)
        # self.SRB2_bn3 = nn.BatchNorm2d(self.scale)
        # self.pPLIF6 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB2_conv3 = nn.Conv2d(self.scale, self.scale*4, 1, stride=1, padding=0, bias=False)
        self.SRB2_skip = nn.Sequential(nn.BatchNorm2d(self.scale*4))   
        self.SRB2_feedback = nn.Conv2d(self.scale, self.scale*4, 1, stride=1, bias=False)   

        self.SRB3_bn1 = nn.BatchNorm2d(self.scale*4)
        self.pPLIF5 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_conv1 = nn.Conv2d(self.scale*4, self.scale*2, 3, stride=2, padding=1, bias=False)
        # self.SRB3_conv1 = nn.Conv2d(self.scale*4, self.scale*2, 3, stride=1, padding=1, bias=False)
        self.SRB3_bn2 = nn.BatchNorm2d(self.scale*2)
        self.pPLIF6 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_conv2 = nn.Conv2d(self.scale*2, self.scale*2*4, 3, stride=1, padding=1, bias=False)
        # self.SRB3_bn3 = nn.BatchNorm2d(self.scale*2)
        # self.pPLIF9 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB3_conv3 = nn.Conv2d(self.scale*2, self.scale*2*4, 1, stride=1, padding=0, bias=False)
        self.SRB3_skip = nn.Conv2d(self.scale*4, self.scale*2*4, 1, stride=2, bias=False)
        self.SRB3_feedback = nn.Conv2d(self.scale*2, self.scale*4, 1, stride=1, bias=False)   


        self.SRB4_bn1 = nn.BatchNorm2d(self.scale*2*4)
        self.pPLIF7 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_conv1 = nn.Conv2d(self.scale*2*4, self.scale*2, 3, stride=1, padding=1, bias=False)
        self.SRB4_bn2 = nn.BatchNorm2d(self.scale*2)
        self.pPLIF8 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_conv2 = nn.Conv2d(self.scale*2, self.scale*2*4 , 3, stride=1, padding=1, bias=False)
        # self.SRB4_bn3 = nn.BatchNorm2d(self.scale*2)
        # self.pPLIF12 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB4_conv3 = nn.Conv2d(self.scale*2, self.scale*2, 1, stride=1, padding=0, bias=False)
        self.SRB4_skip = nn.Sequential(nn.BatchNorm2d(self.scale*2*4)) 
        self.SRB4_feedback = nn.Conv2d(self.scale*2, self.scale*2*4, 1, stride=1, bias=False)   

        self.SRB5_bn1 = nn.BatchNorm2d(self.scale*2*4)
        self.pPLIF9 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_conv1 = nn.Conv2d(self.scale*2*4, self.scale*4, 3, stride=2, padding=1, bias=False)
        # self.SRB5_conv1 = nn.Conv2d(self.scale*2*4, self.scale*4, 3, stride=1, padding=1, bias=False)
        self.SRB5_bn2 = nn.BatchNorm2d(self.scale*4)
        self.pPLIF10 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_conv2 = nn.Conv2d(self.scale*4, self.scale*4*4 , 3, stride=1, padding=1, bias=False)
        # self.SRB5_bn3 = nn.BatchNorm2d(self.scale*4)
        # self.pPLIF15 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB5_conv3 = nn.Conv2d(self.scale*4, self.scale*4*4, 1, stride=1, padding=0, bias=False)
        self.SRB5_skip = nn.Conv2d(self.scale*2*4, self.scale*4*4, 1, stride=2, bias=False)
        self.SRB5_feedback = nn.Conv2d(self.scale*4, self.scale*2*4, 1, stride=1, bias=False)   

        self.SRB6_bn1 = nn.BatchNorm2d(self.scale*4*4)
        self.pPLIF11 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_conv1 = nn.Conv2d(self.scale*4*4, self.scale*4, 3, stride=1, padding=1, bias=False)
        self.SRB6_bn2 = nn.BatchNorm2d(self.scale*4)
        self.pPLIF12 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_conv2 = nn.Conv2d(self.scale*4, self.scale*4*4, 3, stride=1, padding=1, bias=False)
        # self.SRB6_bn3 = nn.BatchNorm2d(self.scale*4)
        # self.pPLIF18 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB6_conv3 = nn.Conv2d(self.scale*4, self.scale*4*4, 1, stride=1, padding=0, bias=False)
        self.SRB6_skip = nn.Sequential(nn.BatchNorm2d(self.scale*4*4)) 
        self.SRB6_feedback = nn.Conv2d(self.scale*4*4, self.scale*4*4, 1, stride=1, bias=False)   

        self.SRB7_bn1 = nn.BatchNorm2d(self.scale*4*4)
        self.pPLIF13 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_conv1 = nn.Conv2d(self.scale*4*4, self.scale*8, 3, stride=2, padding=1, bias=False)
        # self.SRB7_conv1 = nn.Conv2d(self.scale*4*4, self.scale*8, 3, stride=1, padding=1, bias=False)
        self.SRB7_bn2 = nn.BatchNorm2d(self.scale*8)
        self.pPLIF14 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_conv2 = nn.Conv2d(self.scale*8, self.scale*8*4, 3, stride=1, padding=1, bias=False)
        # self.SRB7_bn3 = nn.BatchNorm2d(self.scale*8)
        # self.pPLIF21 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB7_conv3 = nn.Conv2d(self.scale*8, self.scale*8*4, 1, stride=1, padding=0, bias=False)
        self.SRB7_skip = nn.Conv2d(self.scale*4*4, self.scale*8*4, 1, stride=2, bias=False)
        self.SRB7_feedback = nn.Conv2d(self.scale*8, self.scale*4*4, 1, stride=1, bias=False)   

        self.SRB8_bn1 = nn.BatchNorm2d(self.scale*8*4)
        self.pPLIF15 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_conv1 = nn.Conv2d(self.scale*8*4, self.scale*8, 3, stride=1, padding=1, bias=False)
        self.SRB8_bn2 = nn.BatchNorm2d(self.scale*8)
        self.pPLIF16 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_conv2 = nn.Conv2d(self.scale*8, self.scale*8*4, 3, stride=1, padding=1, bias=False)
        # self.SRB8_bn3 = nn.BatchNorm2d(self.scale*8)
        # self.pPLIF24 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        # self.dropout = layer.Dropout(0)
        # self.SRB8_conv3 = nn.Conv2d(self.scale*8, self.scale*8*4, 1, stride=1, padding=0, bias=False)
        self.SRB8_skip = nn.Sequential(nn.BatchNorm2d(self.scale*8*4)) 
        self.SRB8_feedback = nn.Conv2d(self.scale*8, self.scale*8*4, 1, stride=1, bias=False)   

        # fc

        self.bn1 = nn.BatchNorm2d(self.scale*8*4)
        self.pPLIF17 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.pool1 = nn.AvgPool2d(4)

        self.fc1 = nn.Linear(self.scale*8 * 4, 100, bias=False)
        self.pPLIF18 = pPLIF_Node(surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))

        self.boost1 = nn.AvgPool1d(10, 10)

        self.pPLI = pPLI_Node(decay_acc=init_decay_acc, surrogate_function=surrogate.AccAlwaysGradCall())

        # self.tau_vector = nn.Parameter(torch.ones(9, dtype=torch.float)*init_tau)
        self.tau_vector = nn.Parameter(torch.tensor([1.6541, 0.9080, 0.9243, 0.6587, 1.0764, 0.9103, 0.9568, 0.9435, 1.0807, 0.9568, 0.9568, 0.9568, 0.9568, 0.9568, 0.9568, 0.9568, 0.9568, 0.9568]))
        self.tau_vector.to("cuda" if torch.cuda.is_available() else "cpu")

        # self.acc_tau = nn.Parameter(torch.ones(1, dtype=torch.float)*init_acc_tau)
        self.acc_tau = nn.Parameter(torch.tensor([5.6898]))
        self.acc_tau.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_1_mem = srb1_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_2_mem = srb1_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb2_1_mem = srb2_1_spike = torch.zeros(batch_size, self.scale*4, 32, 32, device=self.device)
        srb2_2_mem = srb2_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb3_1_mem = srb3_1_spike = torch.zeros(batch_size, self.scale*4, 32, 32, device=self.device)
        srb3_2_mem = srb3_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb4_1_mem = srb4_1_spike = torch.zeros(batch_size, self.scale*2*4, 16, 16, device=self.device)
        srb4_2_mem = srb4_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device) 
        srb5_1_mem = srb5_1_spike = torch.zeros(batch_size, self.scale*2*4, 16, 16, device=self.device)
        srb5_2_mem = srb5_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb6_1_mem = srb6_1_spike = torch.zeros(batch_size, self.scale*4*4, 8, 8, device=self.device)
        srb6_2_mem = srb6_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb7_1_mem = srb7_1_spike = torch.zeros(batch_size, self.scale*4*4, 8, 8, device=self.device)
        srb7_2_mem = srb7_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device) 
        srb8_1_mem = srb8_1_spike = torch.zeros(batch_size, self.scale*8*4, 4, 4, device=self.device)
        srb8_2_mem = srb8_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)        
        h1_mem = h1_spike = torch.zeros(batch_size, self.scale*8*4, 4, 4, device=self.device) 
        h2_mem = h2_spike = torch.zeros(batch_size, 100, device=self.device)
        boost1 = torch.zeros(batch_size, 10, device=self.device)
        acc_mem = torch.zeros(batch_size, 10, device=self.device)

        c1_mem.fill_(0.5)
        srb1_1_mem.fill_(0.5)
        srb1_2_mem.fill_(0.5)
        srb2_1_mem.fill_(0.5)
        srb2_2_mem.fill_(0.5)
        srb3_1_mem.fill_(0.5)
        srb3_2_mem.fill_(0.5)
        srb4_1_mem.fill_(0.5)
        srb4_2_mem.fill_(0.5)
        srb5_1_mem.fill_(0.5)
        srb5_2_mem.fill_(0.5)
        srb6_1_mem.fill_(0.5)
        srb6_2_mem.fill_(0.5)
        srb7_1_mem.fill_(0.5)
        srb7_2_mem.fill_(0.5)
        srb8_1_mem.fill_(0.5)
        srb8_2_mem.fill_(0.5)
        h1_mem.fill_(0.5)
        h2_mem.fill_(0.5)

        decay_vector = torch.sigmoid(self.tau_vector)
        acc_decay = torch.sigmoid(self.acc_tau)

        for step in range(self.num_steps-1):
            with torch.no_grad():
                c1_out = self.conv1(x)

                #

                srb1_1_mem, srb1_1_spike = self.pPLIF1(srb1_1_mem.detach(), srb1_1_spike.detach(), decay_vector[0], self.SRB1_bn1(c1_out))

                srb1_2_mem, srb1_2_spike = self.pPLIF2(srb1_2_mem.detach(), srb1_2_spike.detach(), decay_vector[1], self.SRB1_bn2(self.SRB1_conv1(srb1_1_spike)))
                
                srb1_out = self.SRB1_conv2(srb1_2_spike) +self.SRB1_skip(c1_out)

                srb2_1_mem, srb2_1_spike = self.pPLIF3(srb2_1_mem.detach(), srb2_1_spike.detach(), decay_vector[2], self.SRB2_bn1(srb1_out))

                srb2_2_mem, srb2_2_spike = self.pPLIF4(srb2_2_mem.detach(), srb2_2_spike.detach(), decay_vector[3], self.SRB2_bn2(self.SRB2_conv1(srb2_1_spike)))
                
                srb2_out = self.SRB2_conv2(srb2_2_spike) + self.SRB2_skip(srb1_out)

                srb3_1_mem, srb3_1_spike = self.pPLIF5(srb3_1_mem.detach(), srb3_1_spike.detach(), decay_vector[4], self.SRB3_bn1(srb2_out))

                srb3_2_mem, srb3_2_spike = self.pPLIF6(srb3_2_mem.detach(), srb3_2_spike.detach(), decay_vector[5], self.SRB3_bn2(self.SRB3_conv1(srb3_1_spike)))
                
                srb3_out = self.SRB3_conv2(srb3_2_spike) +self.SRB3_skip(srb2_out)

                srb4_1_mem, srb4_1_spike = self.pPLIF7(srb4_1_mem.detach(), srb4_1_spike.detach(), decay_vector[6], self.SRB4_bn1(srb3_out))

                srb4_2_mem, srb4_2_spike = self.pPLIF8(srb4_2_mem.detach(), srb4_2_spike.detach(), decay_vector[7], self.SRB4_bn2(self.SRB4_conv1(srb4_1_spike)))
                
                srb4_out = self.SRB4_conv2(srb4_2_spike) +self.SRB4_skip(srb3_out)

                srb5_1_mem, srb5_1_spike = self.pPLIF9(srb5_1_mem.detach(), srb5_1_spike.detach(), decay_vector[8], self.SRB5_bn1(srb4_out))

                srb5_2_mem, srb5_2_spike = self.pPLIF10(srb5_2_mem.detach(), srb5_2_spike.detach(), decay_vector[9], self.SRB5_bn2(self.SRB5_conv1(srb5_1_spike)))
                
                srb5_out = self.SRB5_conv2(srb5_2_spike) +self.SRB5_skip(srb4_out)

                srb6_1_mem, srb6_1_spike = self.pPLIF11(srb6_1_mem.detach(), srb6_1_spike.detach(), decay_vector[10], self.SRB6_bn1(srb5_out))

                srb6_2_mem, srb6_2_spike = self.pPLIF12(srb6_2_mem.detach(), srb6_2_spike.detach(), decay_vector[11], self.SRB6_bn2(self.SRB6_conv1(srb6_1_spike)))
                
                srb6_out = self.SRB6_conv2(srb6_2_spike) +self.SRB6_skip(srb5_out)

                srb7_1_mem, srb7_1_spike = self.pPLIF13(srb7_1_mem.detach(), srb7_1_spike.detach(), decay_vector[12], self.SRB7_bn1(srb6_out))

                srb7_2_mem, srb7_2_spike = self.pPLIF14(srb7_2_mem.detach(), srb7_2_spike.detach(), decay_vector[13], self.SRB7_bn2(self.SRB7_conv1(srb7_1_spike)))
                
                srb7_out = self.SRB7_conv2(srb7_2_spike) +self.SRB7_skip(srb6_out)

                srb8_1_mem, srb8_1_spike = self.pPLIF15(srb8_1_mem.detach(), srb8_1_spike.detach(), decay_vector[14], self.SRB8_bn1(srb7_out))

                srb8_2_mem, srb8_2_spike = self.pPLIF16(srb8_2_mem.detach(), srb8_2_spike.detach(), decay_vector[15], self.SRB8_bn2(self.SRB8_conv1(srb8_1_spike)))
                
                srb8_out = self.SRB8_conv2(srb8_2_spike) +self.SRB8_skip(srb7_out)

                h1_mem, h1_spike = self.pPLIF17(h1_mem.detach(), h1_spike.detach(), decay_vector[16], self.bn1(srb8_out))

                h2_mem, h2_spike = self.pPLIF18(h2_mem.detach(), h2_spike.detach(), decay_vector[17], self.fc1(self.pool1(h1_spike).view(batch_size, -1)))
                # h2_mem, h2_spike = self.pPLIF18(h2_mem.detach(), h2_spike.detach(), decay_vector[17], self.fc1(self.pool2(self.pool1(h1_spike)).view(batch_size, -1)))

                boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)

                acc_mem = self.pPLI(acc_mem.detach(), acc_decay, boost1)

        c1_out = self.conv1(x)

        #

        srb1_1_mem, srb1_1_spike = self.pPLIF1(srb1_1_mem.detach(), srb1_1_spike.detach(), decay_vector[0], self.SRB1_bn1(c1_out))

        srb1_2_mem, srb1_2_spike = self.pPLIF2(srb1_2_mem.detach(), srb1_2_spike.detach(), decay_vector[1], self.SRB1_bn2(self.SRB1_conv1(srb1_1_spike)))
        
        srb1_out = self.SRB1_conv2(srb1_2_spike) +self.SRB1_skip(c1_out)

        srb2_1_mem, srb2_1_spike = self.pPLIF3(srb2_1_mem.detach(), srb2_1_spike.detach(), decay_vector[2], self.SRB2_bn1(srb1_out))

        srb2_2_mem, srb2_2_spike = self.pPLIF4(srb2_2_mem.detach(), srb2_2_spike.detach(), decay_vector[3], self.SRB2_bn2(self.SRB2_conv1(srb2_1_spike)))
        
        srb2_out = self.SRB2_conv2(srb2_2_spike) + self.SRB2_skip(srb1_out)

        srb3_1_mem, srb3_1_spike = self.pPLIF5(srb3_1_mem.detach(), srb3_1_spike.detach(), decay_vector[4], self.SRB3_bn1(srb2_out))

        srb3_2_mem, srb3_2_spike = self.pPLIF6(srb3_2_mem.detach(), srb3_2_spike.detach(), decay_vector[5], self.SRB3_bn2(self.SRB3_conv1(srb3_1_spike)))
        
        srb3_out = self.SRB3_conv2(srb3_2_spike) +self.SRB3_skip(srb2_out)

        srb4_1_mem, srb4_1_spike = self.pPLIF7(srb4_1_mem.detach(), srb4_1_spike.detach(), decay_vector[6], self.SRB4_bn1(srb3_out))

        srb4_2_mem, srb4_2_spike = self.pPLIF8(srb4_2_mem.detach(), srb4_2_spike.detach(), decay_vector[7], self.SRB4_bn2(self.SRB4_conv1(srb4_1_spike)))
        
        srb4_out = self.SRB4_conv2(srb4_2_spike) +self.SRB4_skip(srb3_out)

        srb5_1_mem, srb5_1_spike = self.pPLIF9(srb5_1_mem.detach(), srb5_1_spike.detach(), decay_vector[8], self.SRB5_bn1(srb4_out))

        srb5_2_mem, srb5_2_spike = self.pPLIF10(srb5_2_mem.detach(), srb5_2_spike.detach(), decay_vector[9], self.SRB5_bn2(self.SRB5_conv1(srb5_1_spike)))
        
        srb5_out = self.SRB5_conv2(srb5_2_spike) +self.SRB5_skip(srb4_out)

        srb6_1_mem, srb6_1_spike = self.pPLIF11(srb6_1_mem.detach(), srb6_1_spike.detach(), decay_vector[10], self.SRB6_bn1(srb5_out))

        srb6_2_mem, srb6_2_spike = self.pPLIF12(srb6_2_mem.detach(), srb6_2_spike.detach(), decay_vector[11], self.SRB6_bn2(self.SRB6_conv1(srb6_1_spike)))
        
        srb6_out = self.SRB6_conv2(srb6_2_spike) +self.SRB6_skip(srb5_out)

        srb7_1_mem, srb7_1_spike = self.pPLIF13(srb7_1_mem.detach(), srb7_1_spike.detach(), decay_vector[12], self.SRB7_bn1(srb6_out))

        srb7_2_mem, srb7_2_spike = self.pPLIF14(srb7_2_mem.detach(), srb7_2_spike.detach(), decay_vector[13], self.SRB7_bn2(self.SRB7_conv1(srb7_1_spike)))
        
        srb7_out = self.SRB7_conv2(srb7_2_spike) +self.SRB7_skip(srb6_out)

        srb8_1_mem, srb8_1_spike = self.pPLIF15(srb8_1_mem.detach(), srb8_1_spike.detach(), decay_vector[14], self.SRB8_bn1(srb7_out))

        srb8_2_mem, srb8_2_spike = self.pPLIF16(srb8_2_mem.detach(), srb8_2_spike.detach(), decay_vector[15], self.SRB8_bn2(self.SRB8_conv1(srb8_1_spike)))
        
        srb8_out = self.SRB8_conv2(srb8_2_spike) +self.SRB8_skip(srb7_out)

        h1_mem, h1_spike = self.pPLIF17(h1_mem.detach(), h1_spike.detach(), decay_vector[16], self.bn1(srb8_out))

        h2_mem, h2_spike = self.pPLIF18(h2_mem.detach(), h2_spike.detach(), decay_vector[17], self.fc1(self.pool1(h1_spike).view(batch_size, -1)))
        # h2_mem, h2_spike = self.pPLIF18(h2_mem.detach(), h2_spike.detach(), decay_vector[17], self.fc1(self.pool2(self.pool1(h1_spike)).view(batch_size, -1)))

        boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)

        acc_mem = self.pPLI(acc_mem.detach(), acc_decay, boost1)

        return acc_mem, self.num_steps
        # return next - softmax and cross-entropy loss