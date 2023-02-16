from faster_rcnn import *
from train import show_image


def rescale_boxes(boxes, image):
    im_h, im_w = image.height, image.width

    return boxes * [
        im_w / 512,
        im_h / 512,
        im_w / 512,
        im_h / 512,
    ]


if __name__ == "__main__":
    idx = 120
    dataset_path = Path("C:/Users/fayaz/NTNU/Norsvin/Norsvin/")
    val_images_path = dataset_path / "val/images"
    val_anns_path = dataset_path / "val/annotations"

    dataset = PigsDataset(val_images_path, val_anns_path)

    model = FasterRCNNModel(num_classes=2)
    model.load_state_dict(torch.load("results/weights/faster_rcnn_weights"))
    model.eval()

    image, target = dataset[idx]

    out = model([image])

    image = Image.open(dataset.image_paths[idx])
    boxes = np.array(out[0]["boxes"].detach().cpu())
    boxes = rescale_boxes(boxes, image)

    show_image(image, boxes)
