def visualize_training_data(train_dir, num_images=6):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    gt_dir = os.path.join(train_dir, "GT")
    input_dir = os.path.join(train_dir, "Input")
    files = os.listdir(input_dir)[:num_images]
    combined_images = []
    for f in files:
        input_path = os.path.join(input_dir, f)
        gt_path = os.path.join(gt_dir, f)
        if os.path.exists(input_path) and os.path.exists(gt_path):
            input_img = Image.open(input_path)
            gt_img = Image.open(gt_path)
            # Concatenate images horizontally
            new_img = Image.new('RGB', (input_img.width+gt_img.width, input_img.height))
            new_img.paste(input_img, (0,0))
            new_img.paste(gt_img, (input_img.width,0))
            combined_images.append(new_img)
    cols = 3
    rows = (len(combined_images) + cols - 1) // cols
    plt.figure(figsize=(cols*6, rows*6))
    for idx, img in enumerate(combined_images):
        plt.subplot(rows, cols, idx+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_unpaired(folder, num_images=10):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    input_dir = os.path.join(folder, "Input")
    files = os.listdir(input_dir)[:num_images]
    images = [Image.open(os.path.join(input_dir, f)) for f in files]
    cols = 5
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols*6, rows*6))
    for idx, img in enumerate(images):
        plt.subplot(rows, cols, idx+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()