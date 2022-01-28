from torch.utils.data import Dataset
import os
import json


class VQADataset(Dataset):
    def __init__(self, root_dir, mode="train", numbers_only=False):
        self.mode = mode
        self.root_dir = root_dir
        """
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
        )
        """
        # questions        
        if mode == "train":
            self.questions_path = os.path.join(self.root_dir,'tools/data/vqa/OpenEnded_mscoco_train2014_questions.json')
            self.annotations_path = os.path.join(self.root_dir, 'tools/data/vqa/mscoco_train2014_annotations.json')
        elif mode == "val":
            self.questions_path = os.path.join(self.root_dir, 'tools/data/vqa/OpenEnded_mscoco_val2014_questions.json')
            self.annotations_path = os.path.join(self.root_dir, 'tools/data/vqa/mscoco_val2014_annotations.json')

        with open(self.questions_path) as question_file:
            self.questions = json.load(question_file)["questions"]
        with open(self.annotations_path) as annotations_file:
            self.annotations = json.load(annotations_file)["annotations"]
        
        
        if numbers_only:
            self.questions = [q for q in self.questions if q["question"].lower().startswith("what number")]
        else:
            #self.questions = [q for q in self.questions if q["question"].lower().startswith("what color")]
            #self.questions = [q for q in self.questions if q["question"].lower().startswith("what shape")] 
            self.questions = [q for q in self.questions if q["question"].lower().startswith("what color") or q["question"].lower().startswith("what shape")]
        print(self.mode, len(self.questions))

def main():
    train_dataset = VQADataset(root_dir="./", mode="train")
    val_dataset = VQADataset(root_dir="./", mode="val")
if __name__ == "__main__":
    main()
