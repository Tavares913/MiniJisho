from torch.utils.data import Dataset, DataLoader

class Translation(Dataset):
    def __init__(self, english_sentences, japanese_sentences):
        if (len(english_sentences) != len(japanese_sentences)):
            raise Exception("The number of english sentences doesn't match the number of japanese ones")
        self.english_sentences = english_sentences
        self.japanese_sentences = japanese_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, index):
        return self.english_sentences[index], self.japanese_sentences[index]
