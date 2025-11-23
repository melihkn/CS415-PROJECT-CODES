from helpers.evaluate_on_loader import evaluate_on_loader
from data.dataset import get_dataloader
import torch
import os
from models.hfanet import HFANet
from models.hfanet import HFANet_timm
from models.HDANet.hdanet import HDANet
from models.stanet import STANet

def test(args):

    threshold = args.threshold  # binarization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split="test")

    # Model selection
    if args.model == "hfanet":
        model = HFANet(encoder_name=args.backbone, classes=1, pretrained=None)
    elif args.model == "hfanet_timm":
        model = HFANet_timm(encoder_name=args.backbone, classes=1, pretrained=None)
    elif args.model == "hdanet":
        model = HDANet(n_classes=1, pretrained=None)
    elif args.model == "stanet":
        model = STANet(backbone_name=args.backbone, classes=1, pretrained=None)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Model to device
    model.to(device)
    print(f"Model {args.model} initialized with backbone {args.backbone}.")

    # TEST
    if test_loader is not None:
        best_model_path = os.path.join("checkpoints", f"best_model_{args.model}.pth")
        # Load best model weights for testing
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Loaded best model weights for testing.")

        evaluate_on_loader(
            model,
            test_loader,
            device,
            threshold=threshold,
            save_pr_curve_path=f"results/pr_curve_{args.model}.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building Change Detection Testing")

    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--model", type=str, default="hfanet",
                        choices=["hfanet", "hfanet_timm", "hdanet", "stanet"])
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    test(args)
