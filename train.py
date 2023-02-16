from pathlib import Path
from objdetecteval.metrics.coco_metrics import get_coco_stats
from matplotlib import patches
from pytorch_lightning import Trainer
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.soft_nms import soft_nms
from effdet.smart_nms import smart_nms
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from typing import List
from fastcore.dispatch import typedispatch
from ensemble_boxes import ensemble_boxes_wbf
from effdet.efficientdet import HeadNet

import os

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
import torch

architecture = "tf_efficientdet_d0"
img_size = (512, 512)


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def get_pascal_bbox(bbox):
    xmin, ymin = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    xmax = xmin + width
    ymax = ymin + height
    return [xmin, ymin, xmax, ymax]


def get_pascal_bboxes(bboxes):
    # coco -> pascal
    out = []
    for bbox in bboxes:
        out.append(get_pascal_bbox(bbox))
    return out


def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    confidences=None,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for i, bbox in enumerate(bboxes):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        if confidences:
            rx, ry = rect_2.get_xy()
            plot_ax.annotate(
                np.round(confidences[i], 2), (rx, ry), color="red", weight="bold"
            )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()


class PigsDatasetAdapter:
    def __init__(self, images_path, anns_path):
        self.image_paths = [images_path / path for path in os.listdir(images_path)]
        self.coco_anns = [
            COCO(anns_path / (path[:-4] + ".json")) for path in os.listdir(images_path)
        ]

    def __len__(self):
        return len(self.image_paths)

    def get_image_and_labels_by_idx(self, index):
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
        labels = np.ones(len(anns))
        labels = np.delete(labels, bad_anns, 0)

        return image, boxes, labels, index

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    # efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
    #     name="tf_efficientnetv2_l",
    #     backbone_name="tf_efficientnetv2_l",
    #     backbone_args=dict(drop_path_rate=0.2),
    #     num_classes=num_classes,
    #     url="",
    # )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = nn.Identity()
    net.box_net = nn.Identity()
    # net.load_state_dict(
    #     torch.load("results/weights/barlow_pretrained_weights_backbone")
    # )
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    net.box_net = HeadNet(config, num_outputs=4)
    return DetBenchTrain(net, config)


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


class EfficientDetDataset(Dataset):
    def __init__(
        self,
        dataset_adaptor,
        transforms=get_valid_transforms(target_img_size=img_size[0]),
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


class EfficientDetDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms=get_train_transforms(target_img_size=img_size[0]),
        valid_transforms=get_valid_transforms(target_img_size=img_size[0]),
        num_workers=4,
        batch_size=8,
    ):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids


def run_wbf(
    predictions, image_size=img_size[0], iou_thr=0.44, skip_box_thr=0.43, weights=None
):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


def run_soft_nms(predictions, iou_thr=0.44, skip_box_thr=0.12, sigma=0.5):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["classes"]

        idxs_out, scores_out = soft_nms(
            boxes,
            scores,
            iou_threshold=iou_thr,
            score_threshold=skip_box_thr,
            sigma=sigma,
        )
        bboxes.append(boxes[idxs_out].tolist())
        confidences.append(scores_out.tolist())
        class_labels.append(labels[idxs_out].tolist())

    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        img_size=img_size[0],
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        iou_threshold=0.44,
        sigma=0.5,
        inference_transforms=get_valid_transforms(target_img_size=img_size[0]),
        model_architecture="tf_efficientnetv2_l",
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.iou_threshold = iou_threshold
        self.sigma = sigma
        self.inference_tfms = inference_transforms

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.lr,
        #     momentum=0.9,
        #     weight_decay=4e-5,
        #     nesterov=True,
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=5, eta_min=0.001, verbose=True
        # )
        # return [optimizer], [scheduler]
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log(
            "valid_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_class_loss",
            logging_losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_box_loss",
            logging_losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": outputs["loss"], "batch_predictions": batch_predictions}

    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: np.ndarray):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor[np.newaxis, :]

        image_sizes = [(image.shape[0], image.shape[1]) for image in images_tensor]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=image,
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images_tensor
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader
        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
            images_tensor.shape[-1] != self.img_size
            or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_confidences, predicted_class_labels

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = run_soft_nms(
            predictions,
            iou_thr=self.iou_threshold,
            sigma=self.sigma,
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        detections = detections.detach().cpu()
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes


def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    predicted_class_confidences,
    actual_bboxes,
    draw_bboxes_fn=draw_pascal_voc_bboxes,
    figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Actual")
    ax2.imshow(image)
    ax2.set_title("Prediction")

    draw_bboxes_fn(ax1, [], None)
    draw_bboxes_fn(ax2, predicted_bboxes, predicted_class_confidences)

    plt.savefig("./results/images/result.png")
    plt.show()


from fastcore.basics import patch


@patch
def aggregate_prediction_outputs(self: EfficientDetModel, outputs):
    detections = torch.cat(
        [output["batch_predictions"]["predictions"] for output in outputs]
    )

    image_ids = []
    targets = []
    for output in outputs:
        batch_predictions = output["batch_predictions"]
        image_ids.extend(batch_predictions["image_ids"])
        targets.extend(batch_predictions["targets"])

    (
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    ) = self.post_process_detections(detections)

    return (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    )


@patch
def validation_epoch_end(self: EfficientDetModel, outputs):
    """Compute and log training loss and accuracy at the epoch level."""

    validation_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()

    (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    ) = self.aggregate_prediction_outputs(outputs)

    truth_image_ids = [target["image_id"].detach().item() for target in targets]
    truth_boxes = [
        target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
    ]  # convert to xyxy for evaluation
    truth_labels = [target["labels"].detach().tolist() for target in targets]

    stats = get_coco_stats(
        prediction_image_ids=image_ids,
        predicted_class_confidences=predicted_class_confidences,
        predicted_bboxes=predicted_bboxes,
        predicted_class_labels=predicted_class_labels,
        target_image_ids=truth_image_ids,
        target_bboxes=truth_boxes,
        target_class_labels=truth_labels,
    )["All"]

    self.log("val_loss", validation_loss_mean)

    return {"val_loss": validation_loss_mean, "metrics": stats}


if __name__ == "__main__":
    dataset_path = Path("C:/Users/fayaz/NTNU/Norsvin/Norsvin/")

    train_images_path = dataset_path / "train/images"
    val_images_path = dataset_path / "val/images"
    train_anns_path = dataset_path / "train/annotations"
    val_anns_path = dataset_path / "val/annotations"

    pigs_train_ds = PigsDatasetAdapter(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapter(val_images_path, val_anns_path)
    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=4,
        batch_size=8,
    )

    model = EfficientDetModel(
        num_classes=1,
        img_size=img_size[0],
        model_architecture=architecture,
        iou_threshold=0.44,
        prediction_confidence_threshold=0.2,
        sigma=0.8,
        learning_rate=0.003,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=30,
        num_sanity_val_steps=1,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        accumulate_grad_batches=3,
    )
    trainer.fit(model, dm)
    torch.save(
        model.state_dict(),
        f"results/weights/norsvin_trained_effdet_{architecture}_{img_size}_no_barlow",
    )
