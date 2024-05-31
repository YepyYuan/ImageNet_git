# split the dataset with 'train_test_split.txt'

def dataset_split(images_txt, split_txt, train_images, test_images):

    is_train_list = []

    with open(split_txt, 'r') as f_split:
        line = f_split.readline()
        while line:
            is_train = int(line.strip().split(' ')[1])
            is_train_list.append(is_train)
            line = f_split.readline()

    with open(images_txt, 'r') as f_imgs:
        lines = f_imgs.readlines()

    train_lines = []
    test_lines = []

    for idx in range(len(lines)):
        line = lines[idx]
        if is_train_list[idx] == 1:
            train_lines.append(line)
        else:
            test_lines.append(line)

    print(len(train_lines))
    print(len(test_lines))

    with open(train_images, 'w') as f_train:
        f_train.writelines(train_lines)
    
    with open(test_images, 'w') as f_test:
        f_test.writelines(test_lines) 

if __name__ == '__main__':
    image_txt = './data/CUB_200_2011/CUB_200_2011/images.txt'
    split_txt = './data/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    train_images = './train_images.txt'
    test_images = './test_images.txt'
    dataset_split(image_txt, split_txt, train_images, test_images)