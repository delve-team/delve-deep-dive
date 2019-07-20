import torch
from torch import nn, optim

from datasets import Food101Dataset
from models import LeNetModel
from trainer import Trainer


batch_size = 10
criterion = nn.CrossEntropyLoss()
model = LeNetModel()
#optimizer = optim.SGD(net.parameters(), lr=0.001)
data = Food101Dataset('tmp')
data.init(selected_classes=['foie_gras', 'tacos'])

trainer = Trainer(model, data)
trainer.train()
trainer.test()

#TrainingDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.training_indices()))
#TestDataLoader = DataLoader(data, batch_size=batch_size, sampler=SubsetRandomSampler(data.test_indices()))


#for index, data in enumerate(TrainingDataLoader):
#    batch_out = net(data)

