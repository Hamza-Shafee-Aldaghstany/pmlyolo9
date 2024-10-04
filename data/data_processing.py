import os
import shutil
from sklearn.model_selection import train_test_split

def load_data(images_dir, labels_dir):
    image_files = []
    label_files = []
    
    # Load images and labels
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):  # Assuming labels are in text files
            label_path = os.path.join(labels_dir, label_file)
            image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))  # Assuming .jpg images
            if os.path.exists(image_path):
                image_files.append(image_path)
                label_files.append(label_path)
    
    return image_files, label_files




def split_data(image_files, label_files, test_size=0.2, val_size=0.1):
    # First split into train+val and test
    assert len(image_files) == len(label_files), "Mismatched lengths: images and labels must have the same number of items."

    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=test_size, random_state=42)
    
    # Then split the train set into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=val_size/(1 - test_size), random_state=42)
    
    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def save_processed_data(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    os.makedirs('./processed/images/train', exist_ok=True)
    os.makedirs('./processed/labels/train', exist_ok=True)
    os.makedirs('./processed/images/valid', exist_ok=True)
    os.makedirs('./processed/labels/valid', exist_ok=True)
    os.makedirs('./processed/images/test', exist_ok=True)
    os.makedirs('./processed/labels/test', exist_ok=True)

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(img, './processed/images/train/')
        shutil.copy(lbl, './processed/labels/train/')
    for img, lbl in zip(val_images, val_labels):
        shutil.copy(img, './processed/images/valid/')
        shutil.copy(lbl, './processed/labels/valid/')
    for img, lbl in zip(test_images, test_labels):
        shutil.copy(img, './processed/images/test/')
        shutil.copy(lbl, './processed/labels/test/')
    datasets = ['train', 'valid', 'test']
    base_path="./processed/images"
    for dataset in datasets:
    # Get the path for the current dataset
        dataset_path = os.path.join(base_path, dataset)
            
            # Create the .txt file for the dataset
        with open(os.path.join("./processed/", f'{dataset}pipe.txt'), 'w') as f:
            # Iterate through each image in the dataset
            for filename in os.listdir(dataset_path):
                if filename.endswith('.jpg'):  # or any image format you expect
                    # Write the relative path to the file
                    f.write(f'./images/{dataset}/{filename}\n')
if __name__ == "__main__":
    images_dir = './raw/images/'  # Specify your images directory
    labels_dir = './raw/labels/'    # Specify your labels directory

    image_files, label_files = load_data(images_dir, labels_dir)
    train_images, val_images,test_images, train_labels, val_labels,test_labels = split_data(image_files, label_files)

    save_processed_data(train_images, train_labels,val_images , val_labels,test_images, test_labels)
