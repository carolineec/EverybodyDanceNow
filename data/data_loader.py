
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataLoaderU(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoaderU
    data_loader = CustomDatasetDataLoaderU()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateTransferDataLoader(opt):
    from data.custom_dataset_data_loader import CustomTransferDatasetDataLoader
    data_loader = CustomTransferDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader