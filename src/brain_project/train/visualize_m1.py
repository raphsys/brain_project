from brain_project.modules.m1_perception import M1PerceptionInherited
from brain_project.utils.visualize import visualize_s1

def main():
    m1 = M1PerceptionInherited(
        k_regions=8,
        pretrained=True,
        freeze_backbone=True,
    )

    visualize_s1(
        model=m1,
        dataset="cifar10",
        data_root="./data",
        index=3,          # change pour voir d'autres images
        img_size=128,
        out_dir="./runs/visualize_s1",
    )

if __name__ == "__main__":
    main()

