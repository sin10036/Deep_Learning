import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)

        Returns:
        class_loss : scalar
        """
        
        ##### CODE #####
        class_loss = F.mse_loss(classes_pred, classes_target, size_average=False)
        
        return class_loss

    def get_regression_loss(self, box_pred_response, box_target_response):   
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        
        """

        
        ##### CODE #####
        

        loss1 = F.mse_loss(box_pred_response[:,0:2],box_target_response[:,0:2],size_average=False)

        loss2 = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)

        reg_loss=loss1+loss2



        return reg_loss
    
    def get_contain_conf_loss(self, box_pred_response, box_target_response_iou):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        contain_loss : scalar
        
        """
        
        ##### CODE #####
       


        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)

        
        return contain_loss
    
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, 30)
        pred_tensor : (tensor) size (batch_size, S , S, 30)
        no_object_mask : (tensor) size (batch_size, S , S, 30)

        Returns:
        no_object_loss : scalar

        Hints:
        1) Create a 2 tensors no_object_prediction and no_object_target which only have the 
        values which have no object. 
        2) Have another tensor no_object_prediction_mask of the same size such that 
        mask with respect to both confidences of bounding boxes set to 1. 
        3) Create 2 tensors which are extracted from no_object_prediction and no_object_target using
        the mask created above to find the loss. 
        """

        no_object_prediction=pred_tensor[no_object_mask].view(-1,30)
        no_object_target=target_tensor[no_object_mask].view(-1,30)
        no_object_prediction_mask = torch.cuda.ByteTensor(no_object_prediction.size())
        no_object_prediction_mask.zero_()
        no_object_prediction_mask[:,4]=1
        no_object_prediction_mask[:,9]=1

        noo_pred_c = no_object_prediction[no_object_prediction_mask] 
        noo_target_c = no_object_target[no_object_prediction_mask]
        no_object_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)
        ##### CODE #####
        
        return no_object_loss
        
    
    
    def find_best_iou_boxes(self, box_target, box_pred):
        """
        Parameters: 
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns: 
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) Set the corresponding contains_object_response_mask of the bounding box with the max iou
        of the 2 bounding boxes of each grid cell to 1.
        3) For finding iou's use the compute_iou function
        4) Before using compute preprocess the bounding box coordinates in such a way that 
        if for a Box b the coordinates are represented by [x, y, w, h] then 
        x, y = x/S - 0.5*w, y/S - 0.5*h ; w, h = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height. 
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        5) Set the confidence of the box_target_iou of the bounding box to the maximum iou
        
        """
        
        ##### CODE #####


        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        return box_target_iou, coo_response_mask
        

        
    
    
    
    def forward(self, pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes
        
        target_tensor: (tensor) size(batchsize,S,S,30)
        
        Returns:
        Total Loss
        '''
        N = pred_tensor.size()[0]
        
        total_loss = None
        
        # Create 2 tensors contains_object_mask and no_object_mask 
        # of size (Batch_size, S, S) such that each value corresponds to if the confidence of having 
        # an object > 0 in the target tensor.


        contains_object_mask=target_tensor[:,:,:,4]>0
        no_object_mask=target_tensor[:,:,:,4]==0
        contains_object_mask = contains_object_mask.unsqueeze(-1).expand_as(target_tensor)
        no_object_mask= no_object_mask.unsqueeze(-1).expand_as(target_tensor)
        
        ##### CODE #####

        # Create a tensor contains_object_pred that corresponds to 
        # to all the predictions which seem to confidence > 0 for having an object
        # Split this tensor into 2 tensors :
        # 1) bounding_box_pred : Contains all the Bounding box predictions of all grid cells of all images
        # 2) classes_pred : Contains all the class predictions for each grid cell of each image
        # Hint : Use contains_object_mask
        
        ##### CODE #####  
        contains_object_pred=pred_tensor[contains_object_mask].view(-1,30)
           
        bounding_box_pred=contains_object_pred[:,0:10].contiguous().view(-1,5)
        classes_pred=contains_object_pred[:,10:]      
        
        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.
        
        ##### CODE #####
        contains_object_target=target_tensor[contains_object_mask].view(-1,30)       
        bounding_box_target=contains_object_target[:,0:10].contiguous().view(-1,5)
        classes_target=contains_object_target[:,10:]

        # Compute the No object loss here
        no_object_loss = self.get_no_object_loss(target_tensor, pred_tensor,no_object_mask)
        ##### CODE #####

        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        ##### CODE #####

        
        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(bounding_box_target,bounding_box_pred)

        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        # Hint : Use contains_object_response_mask
        
        ##### CODE #####
        box_prediction_response = bounding_box_pred[contains_object_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[contains_object_response_mask].view(-1,5)
        box_target_response = bounding_box_target[contains_object_response_mask].view(-1,5)

        # Find the class_loss, containing object loss and regression loss
        
        ##### CODE #####
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)
        regression_loss = self.get_regression_loss(box_prediction_response, box_target_response)
        contain_loss = self.get_contain_conf_loss(box_prediction_response, box_target_response_iou)
        

        total_loss = ((self.l_coord*regression_loss) + (contain_loss) + (self.l_noobj*no_object_loss) +(class_loss))/N
        
        return total_loss