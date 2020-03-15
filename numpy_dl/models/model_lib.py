import numpy_dl as nn


class SimpleNet(nn.Sequencer):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc5 = nn.Linear(200, 1)
        self.sig1 = nn.Sigmoid()

        self.relu1 = nn.ReLU()

    def forward(self, x):
        self.seq = [self.fc1, self.relu1, self.fc5, self.sig1]

        for func in self.seq:
            x = func(x)
        return x


class DemandedNet(nn.Sequencer):

    def __init__(self):
        super(DemandedNet, self).__init__()
        self.fc1 = nn.Linear(2, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc5 = nn.Linear(25, 1)
        # self.sig1 = nn.Sigmoid()
        self.tan1 = nn.Tanh()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        self.seq = [self.fc1, self.relu1, self.fc2, self.fc3, self.fc5, self.tan1]
        for func in self.seq:
            x = func(x)
        return x
