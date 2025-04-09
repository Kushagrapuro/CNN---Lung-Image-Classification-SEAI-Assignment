import os
import shutil
from sklearn.model_selection import train_test_split


original_data_dir = r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\archive\COVID-19_Radiography_Dataset"  
output_dir = r"D:\KUSHAGRA\DATA\K_Works\SEAI Project\dataset"  


classes = ['COVID', 'Viral Pneumonia']  


train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


os.makedirs(output_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


for cls in classes:
    
    cls_images_dir = os.path.join(original_data_dir, cls, 'images')
    print(f"Checking directory for class '{cls}' images: {cls_images_dir}")
    
    if not os.path.exists(cls_images_dir):
        raise FileNotFoundError(f"Directory for class '{cls}' images not found: {cls_images_dir}")
    
    
    all_files = os.listdir(cls_images_dir)
    print(f"All files in '{cls_images_dir}': {all_files}")
    
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = [f for f in all_files if not f.startswith('.') and f.lower().endswith(valid_extensions)]
    print(f"Found {len(images)} valid image files for class '{cls}': {images}")
    
    if len(images) == 0:
        raise ValueError(f"No valid image files found for class '{cls}'. Check the directory and file extensions.")
    
    
    train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
    
    print(f"Class: {cls}")
    print(f"Total Images: {len(images)}")
    print(f"Train: {len(train_images)}, Validation: {len(val_images)}, Test: {len(test_images)}\n")
    
    
    def copy_images(image_list, split):
        for img in image_list:
            src = os.path.join(cls_images_dir, img)  
            dst = os.path.join(output_dir, split, cls, img)
            print(f"Copying: {src} -> {dst}")  
            shutil.copy(src, dst)
    
    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    copy_images(test_images, 'test')

print("Dataset preparation complete!")