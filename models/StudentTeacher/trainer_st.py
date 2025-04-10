import os
import torch
from models.trainer_base import BaseTrainer
from models.teacher import teacherTimm
from models.StudentTeacher.student import studentTimm
from utils.functions import cal_loss
from utils.util import loadWeights

class StTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        
    def load_optim(self):
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 
        
    def load_model(self):
        self.teacher=teacherTimm(backbone_name=self.modelName,out_indices=self.outIndices).to(self.device)
        self.student=studentTimm(backbone_name=self.modelName,out_indices=self.outIndices).to(self.device)

    
    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
      else:
        self.student.eval()  
        
    def infer(self,image):
        self.features_t = self.teacher(image)
        self.features_s=self.student(image)

    def computeLoss(self):
        loss=cal_loss(self.features_s, self.features_t,self.norm)
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")