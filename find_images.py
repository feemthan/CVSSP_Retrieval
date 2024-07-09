import os

def find_images(directory):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'}
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def main():
    directory = input("Enter the directory to search for images: ")
    image_paths = find_images(directory)
    
    with open('image_paths.txt', 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

    print(f"Found {len(image_paths)} images. Paths have been written to 'image_paths.txt'.")

if __name__ == "__main__":
    main()
