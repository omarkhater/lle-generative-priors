def check_train_correspondence(train_dir):
    import os
    gt_dir = os.path.join(train_dir, "GT")
    input_dir = os.path.join(train_dir, "Input")
    gt_images = set(os.listdir(gt_dir))
    input_images = set(os.listdir(input_dir))
    missing = gt_images - input_images
    if missing:
        print(f"#{len(missing)}/{len(gt_images)} GT images do not have corresponding images in Input")
        print("Missing corresponding images in Input\n", missing)
    else:
        print("All GT images have corresponding images in Input.")

def show_stats(base_dir):
    import os
    def count_images(path):
        return len(os.listdir(path)) if os.path.exists(path) else 0
    train_count = count_images(os.path.join(base_dir, "Train", "Input"))
    val_count   = count_images(os.path.join(base_dir, "Val", "Input"))
    test_count  = count_images(os.path.join(base_dir, "Test", "Input"))
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")
    print(f"Test images:  {test_count}")

def rename_files_to_numbers(folder):
    import os
    import re
    for file_name in os.listdir(folder):
        if file_name.lower().endswith('.png'):
            # Remove non-digit prefix from the base filename and reappend .png
            new_name = re.sub(r'^\D+', '', file_name.split('.')[0]) + ".png"
            source = os.path.join(folder, file_name)
            destination = os.path.join(folder, new_name)
            os.rename(source, destination)

