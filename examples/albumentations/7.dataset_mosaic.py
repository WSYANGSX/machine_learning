import torch
from torch.utils.data import DataLoader

from machine_learning.utils.transforms import DEFAULT_YOLO_AUG
from examples.transforms import ImgTransform
from machine_learning.dataset.parsers import YoloParser, YoloParserCfg
from machine_learning.utils.detection import xywh2xyxy
from machine_learning.utils.image import visualize_img_with_bboxes


device = "cuda"


def _prepare_target(si: int, target: torch.Tensor, scale: torch.Tensor):
    """Prepares a batch of images and annotations for validation."""
    idx = target[:, 0] == si
    cls = target[:, 1][idx]
    bbox = target[:, 2:6][idx]
    if len(cls):
        bbox = xywh2xyxy(bbox) * scale  # target boxes
    return {"cls": cls, "bbox": bbox}


def main():
    # Step 1: Parse the data
    tfs = ImgTransform(
        aug_cfg=DEFAULT_YOLO_AUG,
        normalize=True,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_tensor=True,
    )

    parser_cfg = YoloParserCfg(
        dataset_dir="/home/yangxf/WorkSpace/datasets/..datasets/coco",
        labels=True,
        tfs=tfs,
        multiscale=False,
        img_size=640,
    )
    data = YoloParser(parser_cfg).create()

    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
    )

    for _, (imgs, targets) in enumerate(train_loader):
        # Load data
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)  # (img_ids, class_ids, bboxes)

        i = 0
        target = _prepare_target(i, targets, torch.tensor([imgs.size(3), imgs.size(2)] * 2, device=device))
        cls = target["cls"].cpu().numpy()
        bboxes = target["bbox"].cpu().numpy()
        category_id_to_name = {x: str(x) for x in cls}
        visualize_img_with_bboxes(imgs[i].permute(1, 2, 0).cpu().numpy(), bboxes, cls, category_id_to_name)


if __name__ == "__main__":
    main()
