from train import *

if __name__ == "__main__":
    dataset_path = Path("C:/Users/fayaz/NTNU/Norsvin/Norsvin/")

    train_images_path = dataset_path / "train/images"
    val_images_path = dataset_path / "val/images"
    train_anns_path = dataset_path / "train/annotations"
    val_anns_path = dataset_path / "val/annotations"

    pigs_train_ds = PigsDatasetAdapter(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapter(val_images_path, val_anns_path)

    model = EfficientDetModel(
        num_classes=1,
        img_size=img_size[0],
        model_architecture=architecture,
        iou_threshold=0.8,
        prediction_confidence_threshold=0.2,
        sigma=0.8,
    )

    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=2,
        batch_size=8,
    )

    model.load_state_dict(
        torch.load(f"results/weights/norsvin_trained_effdet_{architecture}_{img_size}")
    )
    model.eval()
    image1, truth_bboxes1, _, _ = pigs_val_ds.get_image_and_labels_by_idx(209)
    # image2, truth_bboxes2, _, _ = pigs_val_ds.get_image_and_labels_by_idx(10)
    # image3, truth_bboxes3, _, _ = pigs_val_ds.get_image_and_labels_by_idx(103)
    # image4, truth_bboxes4, _, _ = pigs_val_ds.get_image_and_labels_by_idx(30)
    images = [image1]
    (
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    ) = model.predict(images)
    compare_bboxes_for_image(
        image1,
        predicted_bboxes=predicted_bboxes[0],
        predicted_class_confidences=None,
        actual_bboxes=truth_bboxes1.tolist(),
    )
    # compare_bboxes_for_image(
    #     image2,
    #     predicted_bboxes=predicted_bboxes[1],
    #     predicted_class_confidences=predicted_class_confidences[1],
    #     actual_bboxes=truth_bboxes2.tolist(),
    # )
    # compare_bboxes_for_image(
    #     image3,
    #     predicted_bboxes=predicted_bboxes[2],
    #     predicted_class_confidences=predicted_class_confidences[2],
    #     actual_bboxes=truth_bboxes3.tolist(),
    # )
    # compare_bboxes_for_image(
    #     image4,
    #     predicted_bboxes=predicted_bboxes[3],
    #     predicted_class_confidences=predicted_class_confidences[3],
    #     actual_bboxes=truth_bboxes4.tolist(),
    # )

    # trainer = Trainer(
    #     accelerator="gpu",
    #     devices=[0],
    #     max_epochs=15,
    #     num_sanity_val_steps=1,
    # )
    # trainer.validate(model, dm)
