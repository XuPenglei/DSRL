from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, SimulateDataset
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset.lower() == 'rs_ma':
        train_set = SimulateDataset.SimulateRemoteSensing(
            # X_dir=r'F:\Data\Dream-B\train\image',
            # Xlr_dir=r'F:\Data\Dream-B\train\imageLR',
            # Y_dir=r'F:\Data\Dream-B\train\label',
            X_dir=r'F:\Data\各数据集小测试\Mass\sat',
            Xlr_dir=r'F:\Data\各数据集小测试\Mass\satLR',
            Y_dir=r'F:\Data\各数据集小测试\Mass\map',
            patch_size=512,
            SR=args.SR,
            to_train=True
        )
        val_set = SimulateDataset.SimulateRemoteSensing(
            # X_dir=r'F:\Data\Dream-B\train\image',
            # Xlr_dir=r'F:\Data\Dream-B\train\imageLR',
            # Y_dir=r'F:\Data\Dream-B\train\label',
            X_dir=r'F:\Data\各数据集小测试\Mass\sat',
            Xlr_dir=r'F:\Data\各数据集小测试\Mass\satLR',
            Y_dir=r'F:\Data\各数据集小测试\Mass\map',
            patch_size=512,
            SR=args.SR,
            to_train=False
        )
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset.lower() == 'rs_dreamb':
        train_set = SimulateDataset.SimulateRemoteSensing(
            X_dir=r'/home/tang/桌面/XPL/data/data/train/image',
            Xlr_dir=r'/home/tang/桌面/XPL/data/data/trainLR/image',
            Y_dir=r'/home/tang/桌面/XPL/data/data/train/label',
            patch_size=512,
            SR=args.SR,
            to_train=True
        )
        val_set = SimulateDataset.SimulateRemoteSensing(
            X_dir=r'/home/tang/桌面/XPL/data/data/valid/image',
            Xlr_dir=r'/home/tang/桌面/XPL/data/data/validLR/image',
            Y_dir=r'/home/tang/桌面/XPL/data/data/valid/label',
            patch_size=512,
            SR=args.SR,
            to_train=False
        )
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

