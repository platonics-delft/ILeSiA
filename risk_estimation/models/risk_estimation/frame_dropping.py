
from typing import Any, Iterable, Tuple
import numpy as np
import torch

class FrameDropper():
    def filter_dataset_with_idxs(data: Tuple[Any], idxs: Iterable[int]):
        # Drop points 
        
        data_new = []
        for data_feature in data:
            data_new.append(data_feature[idxs])
        
        return data_new
    
    @classmethod
    def filter_frames(cls, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wrapper that converts torch.Tensors to numpy arrays and finally back to torch.Tensors

        Args:
            data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """        
        
        data_array = []
        for dataitem in data:
            data_array.append(dataitem.cpu().numpy())
        
        data_array = cls._filter_frames(data_array)

        data_torch = []
        for dataitem in data_array:
            data_torch.append(torch.tensor(dataitem, dtype=torch.float32).cuda())
        return data_torch

class NoFrameDroppingPolicy(FrameDropper):
    @classmethod
    def _filter_frames(cls, data: Tuple):
        return data



class OnlyLabelledFramesDroppingPolicy(FrameDropper):
    """If risk flag and safe flag is False, datasample is dropped
    """    
    mintestcut = -1
    maxtestcut = 99999999
    @classmethod
    def _filter_frames(cls, data: Tuple):
        idxs = []
        l = len(data[0])
        for i in range(l):
            if cls.mintestcut < i < cls.maxtestcut:
                if data[1][i] == 1 or data[2][i] == 1:
                    idxs.append(i)
                
        data = cls.filter_dataset_with_idxs(data, idxs)
        return data 

class OnlyLabelledFramesDroppingPolicyRiskPegPick1(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 60
    maxtestcut = 90

class OnlyLabelledFramesDroppingPolicyRiskPegPick2(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 480
    maxtestcut = 510

class OnlyLabelledFramesDroppingPolicyRiskPegDoor1(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 150
    maxtestcut = 240

class OnlyLabelledFramesDroppingPolicyRiskPegDoor2(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 630
    maxtestcut = 710

class OnlyLabelledFramesDroppingPolicyRiskPegPlace1(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 60
    maxtestcut = 150

class OnlyLabelledFramesDroppingPolicyRiskPegPlace2(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 400
    maxtestcut = 460

class OnlyLabelledFramesDroppingPolicyRiskSliderMove1(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 30
    maxtestcut = 90

class OnlyLabelledFramesDroppingPolicyRiskSliderMove2(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 215
    maxtestcut = 245

class OnlyLabelledFramesDroppingPolicyRiskMoveAround1(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 60
    maxtestcut = 120

class OnlyLabelledFramesDroppingPolicyRiskMoveAround2(OnlyLabelledFramesDroppingPolicy):
    mintestcut = 150
    maxtestcut = 210

class OnlyLabelledPhaseDroppingPolicy(FrameDropper):
    """If risk flag and safe flag is False, datasample is dropped
    """    
    @classmethod
    def _filter_frames(cls, data: Tuple):
        idxs = []
        l = len(data[0])
        for i in range(l):
            if data[5][i] != -1:
                idxs.append(i)
                
        data = cls.filter_dataset_with_idxs(data, idxs)
        return data 

class OnlyLabelledFramesDroppingPolicyTest(FrameDropper):
    """If risk flag and safe flag is False, datasample is dropped
    """    
    @classmethod
    def _filter_frames(cls, data: Tuple, testcut=200):
        idxs = []
        l = len(data[0])
        for i in range(l):
            if i > testcut:
                break 
            if data[1][i] == 1 or data[2][i] == 1:
                idxs.append(i)
                
        data = cls.filter_dataset_with_idxs(data, idxs)
        return data 


class ProactiveRiskLabelingDroppingPolicy(FrameDropper): 
    """ Plus all safe indexed """
    """Frames where Risk flags changes 'Extreme Points' are detected, frames near Extreme points ('near_radius') are used

    Args:
        FrameDropper (_type_): _description_

    Returns:
        _type_: _description_
    """    
    near_radius = 10
    limit_interest = None

    @classmethod
    def _filter_frames(cls, data: Tuple):
        risk_flag = data[1]
        safe_flag = data[2]
        # Get points for interest
        idxs = cls.risk_flag_points_of_interest(risk_flag.squeeze())
        
        # Add all safe indexes
        idxs = list(idxs)
        l = len(data[0])
        for i in range(l):
            if safe_flag[i] == 1:
                idxs.append(i)

        return cls.filter_dataset_with_idxs(data, idxs)

    @staticmethod
    def risk_flag_points_of_interest(risk_flag):
        '''  '''
        risk_flag_grad = np.gradient(risk_flag)
        importantidxs = np.where(np.ceil(np.abs(risk_flag_grad)))[0]
        
        importantidxs = importantidxs[:ProactiveRiskLabelingDroppingPolicy.limit_interest]

        near_radius = ProactiveRiskLabelingDroppingPolicy.near_radius

        def is_near(n, importantidx):
            if abs(n - importantidx) < near_radius:
                return True
            else:
                return False

        indx = []
        for n in range(len(risk_flag)):
            for importantidx in importantidxs:
                if is_near(n, importantidx):
                    indx.append(n)
        
        return indx
