from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    # dataset_size = len(dataset)
    print('testing batches = %d' % dataset_size)
    print(len(dataset))
    # create a model
    model = create_model(opt)
    print("created model")
    model.eval()
    print("prepared model for evaluation")
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)
    print("starting evaluation")
    # for i, data in enumerate(islice(dataset, opt.how_many)):
    for i, data in enumerate(dataset):
         print(data["img"].size())
         model.set_input(data)
         model.test()
