import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AOKVQADataset(Dataset):
    def __init__(self, split='train', aokvqa_root='/speedy/DenseInject/datasets/aokvqa', 
                 coco_root='/speedy/DenseInject/datasets/coco'):
        """
        A-OKVQA PyTorch Dataset
        
        Args:
            split (str): 'train' or 'val'
            aokvqa_root (str): Path to A-OKVQA dataset directory
            coco_root (str): Path to COCO dataset directory
        """
        self.split = split
        self.aokvqa_root = aokvqa_root
        self.coco_root = coco_root
        
        # Load JSON data
        json_file = f'aokvqa_v1p0_{split}.json'
        json_path = os.path.join(aokvqa_root, json_file)
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # CLIP normalization constants
        CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        
        # Transform pipeline: resize to 256x256 and normalize with CLIP stats
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        
        # Determine image directory based on split
        if split == 'train':
            self.image_dir = os.path.join(coco_root, 'train2017')
        else:  # val
            self.image_dir = os.path.join(coco_root, 'val2017')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image_id and pad with zeros (COCO naming convention)
        image_id = item['image_id']
        image_filename = f'{image_id:012d}.jpg'  # Pad to 12 digits with zeros
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get question and choices
        question = item['question']
        choices = item['choices']
        correct_choice_idx = item['correct_choice_idx']
        
        return {
            'image': image,
            'question': question,
            'choices': choices,
            'correct_choice_idx': correct_choice_idx,
            'image_id': image_id,
            'question_id': item['question_id']
        }


# Example usage
if __name__ == "__main__":
    # Test the dataset
    train_dataset = AOKVQADataset(split='train')
    val_dataset = AOKVQADataset(split='val')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test loading one sample
    sample = train_dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Question: {sample['question']}")
    print(f"Choices: {sample['choices']}")
    print(f"Correct choice idx: {sample['correct_choice_idx']}")