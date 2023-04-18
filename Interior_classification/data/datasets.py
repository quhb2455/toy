class CustomDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self._df = df
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        img_path = self._df['img_path'].iloc[index]
        
        image = cv2.imread(img_path)
        
        if self.transform :
            image = self.transform(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            return image, label

        else:
            return image#, image_2
        
class 