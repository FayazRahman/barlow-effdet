from torch.utils.data import Dataset
from train import get_valid_transforms, get_train_transforms, get_pascal_bbox
from pycocotools.coco import COCO
from PIL import Image
from pytorch_lightning import LightningModule
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from engine import evaluate
from coco_utils import convert_to_coco_api

import numpy as np
import pytorch_lightning as pl

import torch
import os


def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


class PigsDataset(Dataset):
    def __init__(self, images_path, anns_path, transforms=get_valid_transforms(512)):
        self.image_paths = [images_path / path for path in os.listdir(images_path)]
        self.coco_anns = [
            COCO(anns_path / (path[:-4] + ".json")) for path in os.listdir(images_path)
        ]
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        coco = self.coco_anns[index]
        anns = coco.loadAnns(coco.getAnnIds())
        boxes = np.zeros((len(anns), 4))
        bad_anns = []
        for i, ann in enumerate(anns):
            try:
                boxes[i, :] = get_pascal_bbox(ann["bbox"])
                if np.all(boxes[i, :] == 0):
                    bad_anns.append(i)
                    continue
            except ValueError:
                print("--- BAD BOX ---")
                print("--- image id: {} ---".format(i))
                bad_anns.append(i)
                continue
        boxes = np.delete(boxes, bad_anns, 0)
        labels = np.ones(len(anns), dtype=np.int64)
        labels = np.delete(labels, bad_anns, 0)
        image = np.array(image, dtype=np.float32)

        if self.transforms:
            sample = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = sample["image"]
            boxes = sample["bboxes"]
            labels = sample["labels"]

        iscrowd = torch.zeros(len(anns), dtype=torch.int64)
        image_id = torch.tensor([index])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = target["boxes"]
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = iscrowd

        return image, target

    def __len__(self):
        return len(self.image_paths)


class FasterRCNNModel(LightningModule):
    def __init__(
        self, num_classes=1, learning_rate=0.0002, batch_size=8, num_workers=8
    ):
        super().__init__()

        dataset_path = Path("C:/Users/fayaz/NTNU/Norsvin/Norsvin/")

        train_images_path = dataset_path / "train/images"
        train_anns_path = dataset_path / "train/annotations"

        val_images_path = dataset_path / "val/images"
        val_anns_path = dataset_path / "val/annotations"

        self.model = create_model(num_classes)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = PigsDataset(
            train_images_path, train_anns_path, transforms=get_train_transforms(512)
        )
        self.val_dataset = PigsDataset(
            val_images_path, val_anns_path, transforms=get_valid_transforms(512)
        )
        self.coco = convert_to_coco_api(self.val_dataset)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log(
            "Loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return {"loss": loss}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def on_train_epoch_end(self) -> None:
        evaluate(self.model, self.coco, self.val_dataloader(), self.device)
        self.model.train()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.95,
        #     weight_decay=1e-5,
        #     nesterov=True,
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=6, eta_min=0, verbose=True
        # )
        # return [optimizer], [scheduler]
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    model = FasterRCNNModel(num_classes=2, batch_size=16)
    trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=[0])
    trainer.fit(model)

    torch.save(model.state_dict(), "results/weights/faster_rcnn_weights")

    # model = FasterRCNNModel(num_classes=2, batch_size=16)
    # device = torch.device("cuda:0")
    # model = model.to(device)
    # model.load_state_dict(torch.load("results/weights/faster_rcnn_weights"))
    # evaluate(model.model, model.coco, model.val_dataloader(), model.device)
