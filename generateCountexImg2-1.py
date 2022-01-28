import torch
from utils import save_checkpoint, load_checkpoint
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import config
import json
#from dataset import VQADataset2, get_vqa_classes
from model.generator import Generator
from model.discriminator import Pix2PixDiscriminator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import random

import torchvision

sys.path.append("./VQA/vqa_pytorch/vqa/")
sys.path.append("./VQA/vqa_pytorch/")
sys.path.append('./VQA/vqa_pytorch/vqa/external/skip-thoughts.torch/pytorch/')
sys.path.append('./VQA/vqa_pytorch/vqa/external/pretrained-models.pytorch/')

from vqa_inference import MutanAttInference2

torch.backends.cudnn.benchmark = True
#set seed for reproduceability
torch.manual_seed(222)

normalize_img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
normalize_mask = transforms.Normalize((0.5),(0.5))
trans_to_pil = transforms.ToPILImage()

def norm_tensor(x):
    x = x[0]
    if not all(x == 0):
        x -= x.min()
        x /= x.max()
        x = 2*x - 1
        return(x[None,:])
    else:
        return x
    
def train_fn(disc, gen, vqa_model, loader, dataset, opt_disc, opt_gen, l2, cross_entropy, bce, g_scaler, d_scaler, epoch, L2_LAMBDA, CE_LAMBDA, tb):
    #start the training loop for the given epoch
    loop = tqdm(loader, leave=True)
    for idx, (img, qid) in enumerate(loop):
        img = img.to(config.DEVICE)
        question = dataset.questions[qid.item()]
        
        if not os.path.exists('Questions_and_Answers'):
            os.mkdir('Questions_and_Answers')
        with open("Questions_and_Answers" + f"/question_qid_{qid}.txt", "w") as text_file:
            print(question, file=text_file)
        
        # get attention and logits    
        with torch.no_grad():
            orig_im = img[0].clone()
            orig_img = orig_im #一時的に画像を保存
            orig_im = trans_to_pil(orig_im)
       
        a1, orig_logits, q1, activations = vqa_model.infer(orig_im, question["question"]) 
        with open("Questions_and_Answers" + f"/ans1_qid_{qid}.txt", "w") as text_file:
            print(a1, file=text_file)

        # select only needed classes (colors and shapes)        
        if not torch.argmax(orig_logits).item() in vqa_model.classes:
            print("class {} not in vqa_model.classes".format(torch.argmax(orig_logits).item()))
        orig_logits_new = orig_logits.clone()
        orig_logits_new = torch.Tensor(orig_logits_new.cpu().detach().numpy()[:,vqa_model.classes]).to(config.DEVICE)

        # normalize logits to be between [-1,1]
        orig_logits_new = norm_tensor(orig_logits_new)
        q1 = q1.clone()
        q1 = norm_tensor(q1)

        orig_logits, q1 = orig_logits.to(config.DEVICE), q1.to(config.DEVICE)
        answer = torch.tensor([torch.argmax(orig_logits_new).item()],device=config.DEVICE)

        # compute attention map, foreground object and background
        att, fg_real, bg_real = vqa_model.grad_cam(img[0].cpu(), orig_logits, activations)
        att = att.permute(2,0,1).unsqueeze(0).to(config.DEVICE)
       
        if not os.path.exists('Attention_Maps'):
            os.mkdir('Attention_Maps')
        save_image(att, "Attention_Maps" + f"/att_qid_{qid}.png")
        
        mask = att.clone()
        mask = normalize_mask(mask)
        #mask[att > 0.15] = 0.9
        #mask[att <= 0.2] = 0.2

        # get x_co and the background image
        fg_real = normalize_img(fg_real)
        bg_real = normalize_img(bg_real)
        fg_real, bg_real = fg_real.to(config.DEVICE), bg_real.to(config.DEVICE)
        
        #generate fake image
        with torch.cuda.amp.autocast():                    
            norm_img = normalize_img(img)
            y_fake = gen(torch.cat([fg_real, mask], 1), q1, orig_logits_new)
            #save_image(y_fake, "test.png") 
            #save_image(orig_img, "test_orig.png")

            if not os.path.exists('CountEx_Images'):
                os.mkdir('CountEx_Images')
            #save_image(y_fake, "CountEx_Images" + f"/test_qid_{qid}.png") 
            save_image(orig_img, "CountEx_Images" + f"/test-orig_qid_{qid}.png")
            
            
            #import pdb; pdb.set_trace()
            generated = normalize_img((att * (y_fake * 0.5 + 0.5)) + (1.-att) * img)
                        

        # Train Discriminator
        gen_im = generated[0].clone() 
        #save_image(gen_im, "test_orig.png")
        gen_im = gen_im * 0.5 + 0.5
        save_image(gen_im, "CountEx_Images" + f"/test_qid_{qid}.png")
        gen_im = trans_to_pil(gen_im)
        ###import pdb; pdb.set_trace()

        a2, pred_logits, q2, activations2 = vqa_model.infer(gen_im, question["question"])
        with open("Questions_and_Answers" + f"/ans2_qid_{qid}.txt", "w") as text_file:
            print(a2, file=text_file)

def q2img_path(coco_path, q, split="val"):
    img_id = q['image_id']
    id_len = len(str(img_id))
    img_path = ""
    if split == "val":
        file_name = "val2014/COCO_val2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    elif split == "train":
        file_name = "train2014/COCO_train2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    return img_path

class VQADataset2(Dataset):
    """
    Class to initialize a VQA dataset.

    Parameters
    ----------
        root_dir : string
            Root directory where the data is stored.
        mode : string, optional
            Specifies whether to use the training or validation split of the VQA dataset.
            Has to be in ["train","val"].
            The default value is "train".
        numbers_only : boolean, optional
            If True, only number-based questions are loaded from the dataset.
            If False, color and shape-based questions are loaded as described in the Thesis.
            The default value is False.
    """
    def __init__(self, root_dir, mode="train", numbers_only=False):
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
        )
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
            self.questions = [q for q in self.questions if q["question"].lower().startswith("what color") or q["question"].lower().startswith("what shape")]
        print(len(self.questions))
        
        
        if mode == "train":
            random.Random(4).shuffle(self.questions)
            self.questions = self.questions
        self.createIndex()
        
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question_item = self.questions[index]
        
        img_path = q2img_path(os.path.join(self.root_dir, 'tools/data/coco/'), question_item, split=self.mode)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        #img = img#.permute(1,2,0)
        q_id = torch.tensor([index])
        
        return img, q_id

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.annotations}
        qa =  {ann['question_id']: [] for ann in self.annotations}
        qqa = {ann['question_id']: [] for ann in self.annotations}
        for ann in self.annotations:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA


def get_vqa_classes(dataset, vqa_model):
    classes = []
    for q in dataset.questions:
        qid = q["question_id"]
        ans = dataset.qa[qid]["answers"]
        for a in ans:
            try:
                aid = vqa_model.trainset.ans_to_aid[a["answer"]]
                if aid not in classes:
                    classes.append(aid)
            except:
                pass
    print(f"Number of VQA classes: {len(classes)}")
    return classes


def save_some_examples(gen, val_loader, epoch, folder):
    
    for idx, (img, mask, logits, q_emb, qid) in enumerate(val_loader):
        if idx == 2:
            break
        img, mask, logits, q_emb = img.to(config.DEVICE), mask.to(config.DEVICE), logits.to(config.DEVICE), q_emb.to(config.DEVICE)
        qid = qid.item()
        gen.eval()
        with torch.no_grad():
            y_fake = gen(img, q_emb, logits, mask)
            y_fake = y_fake
            save_image(y_fake, folder + f"/y_gen_qid_{qid}_{epoch}.png")
            if epoch == 0:
                save_image(img, folder + f"/input_qid_{qid}_{epoch}.png")
            if epoch == 1:
                save_image(img, folder + f"/label_qid_{qid}_{epoch}.png")
        gen.train()

def gradient_penalty(critic, real, fake, logits_new, logits_old, device=config.DEVICE):
    BATCH_SIZE, C, H, W = real.shape
    _, ans_shape = logits_new.shape
    alpha = torch.rand((BATCH_SIZE, 1))
    alpha_img = torch.tensor([[[[torch.squeeze(alpha)]]]]).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha_img + fake * (1 - alpha_img)
    alpha_logits = alpha.repeat(1,ans_shape).to(device)
    interpolated_logits = logits_old * alpha_logits + logits_new * (1 - alpha_logits)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images, interpolated_logits)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def main():   
    """Main function to start training procedure of CountEx-VQA."""
    
    # set up folders to save example results during training
    if not os.path.isdir("./evaluation"):
        os.mkdir("./evaluation")
        os.mkdir("./evaluation/training")

    disc = Pix2PixDiscriminator().to(config.DEVICE) 
    gen = Generator(features=64).to(config.DEVICE)
    
    # optimizer for the discriminator
    opt_disc = optim.Adam(disc.parameters(), lr=config.D_LEARNING_RATE, betas=(0.5, 0.999),) #betas and LR based on pix2pix paper
    opt_gen = optim.Adam(gen.parameters(), lr=config.G_LEARNING_RATE, betas=(0.5, 0.999))#, betas=(0.5, 0.999),) # betas and LR based on pix2 pix paper
    L2_LOSS = nn.MSELoss()
    CROSS_ENTROPY = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    #vqa_model = None
   
    train_dataset = VQADataset2(root_dir="./", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    
     #with torch.no_grad():
    vqa_model = MutanAttInference2(dir_logs='./tools/VQA/vqa_pytorch/logs/vqa/mutan_att_trainval', config='./VQA/vqa_pytorch/options/vqa/mutan_att_trainval.yaml')
    classes = get_vqa_classes(train_dataset, vqa_model)
    vqa_model.classes = classes
    vqa_model.model.to(config.DEVICE)
    vqa_model.model.eval() #THIS IS IMPORTANT!
    
    g_scaler = torch.cuda.amp.GradScaler() #faster and uses less VRAM, same results
    d_scaler = torch.cuda.amp.GradScaler()
    
    val_dataset = VQADataset2(root_dir="./", mode="val")
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
 
    #all_losses = {"G_losses": [], "D_losses": [], "D_fake": [], "D_real": []}
    
    for CE_LAMBDA in config.CE_LAMBDA:
        for L2_LAMBDA in config.L2_LAMBDA:
             # tensorboard
            tb = SummaryWriter(log_dir=f'runs/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}_{str(datetime.now())}')
            if not os.path.isdir(f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}"):
                os.mkdir(f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}")
            if config.LOAD_MODEL:
                load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
                load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
               
            loss_file = f"evaluation/losses_{str(datetime.now())}_l2lambda_{L2_LAMBDA}_CE_{CE_LAMBDA}_GLR_{config.G_LEARNING_RATE}_DLR_{config.D_LEARNING_RATE}.json"
            print("TEST!")
            
            for epoch in range(config.NUM_EPOCHS):
                G_losses, D_losses, D_fake_losses, D_real_losses = train_fn(disc, gen, vqa_model, train_loader, train_dataset, opt_disc, opt_gen, L2_LOSS, CROSS_ENTROPY, BCE, g_scaler, d_scaler, epoch, L2_LAMBDA, CE_LAMBDA, tb)
                if config.SAVE_MODEL and epoch % 5 == 0:
                    save_checkpoint(gen, opt_gen, filename=f'gen_L1_{L2_LAMBDA}_CE_{CE_LAMBDA}_epoch_{epoch}.pth.tar')
                    save_checkpoint(disc, opt_disc, filename=f'disc_{L2_LAMBDA}_CE_{CE_LAMBDA}_epoch_{epoch}.pth.tar')
                
                if not os.path.exists("ImageSavingTest"):
                    os.mkdir("ImageSavingTest")
                save_some_examples(gen, val_loader, 1, folder="ImageSavingTest")
                grad = gradient_penalty(critic, real, fake, logit_new, logit_old, device=config.DEVICE)
                if not os.path.exists("Gradients"):
                    os.mkdir("Gradients")
                                
                with open("Gradients" + f"/grad.txt", "w") as text_file:
                    print(grad, file=text_file)

                with open(loss_file, 'w') as f:
                    json.dump(all_losses, f)
            tb.close()
    
if __name__ == "__main__":
    main()
